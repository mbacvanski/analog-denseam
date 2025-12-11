"""
Perform inference on the parity dataset using the pre-trained weights from model_energy_train.py,
but using an explicit ODE solver (diffrax) to integrate the dynamics instead of the unrolled forward-Euler
used during training.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
from dataclasses import dataclass
from typing import Optional

import diffrax
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import Config
from data import load_dataset
from model import infer_forward_euler, logits_from_v
from utils import load_params


BETA = Config.beta
ALPHA = Config.step_size / 10  # initial dt guess for the ODE solver
T_FINAL = float(Config.T_final)
TAU_V = Config.tau_v  # visible neuron time constant (arbitrary units)
TAU_H = Config.tau_h  # hidden neuron time constant (faster relax)


@dataclass
class DynParams:
    XI_attn: jnp.ndarray  # (L, D)
    XI_hopf: jnp.ndarray  # (M, D)
    a: jnp.ndarray  # (D,)
    b_attn: jnp.ndarray  # (L,)
    c: jnp.ndarray  # (M,)
    beta: float  # scalar
    tau_v: float  # scalar
    tau_h: float  # scalar


def make_rhs(P: DynParams):
    """
    Return a vector field fn(t, x, args_unused) -> dx/dt
    with P closed over so we don't have to pass P as `args`.
    """
    XI_attn = P.XI_attn  # (L, D)
    XI_hopf = P.XI_hopf  # (M, D)
    a = P.a  # (D,)
    b_attn = P.b_attn  # (L,)
    c = P.c  # (M,)
    beta = P.beta
    tau_v = P.tau_v
    tau_h = P.tau_h

    L = XI_attn.shape[0]
    D = XI_attn.shape[1]
    M = XI_hopf.shape[0]

    def rhs_closure(t, x, _):
        """
        State x = concat[v (D), h_attn (L), h_hopf (M)].

        Hidden:
            pre_attn = XI_attn @ v + b_attn
            pre_hopf = XI_hopf @ v + c
            dh_attn  = (pre_attn  - h_attn)  / tau_h
            dh_hopf  = (pre_hopf - h_hopf) / tau_h

        Visible:
            p_attn   = softmax(beta * h_attn)
            a_hopf   = relu(h_hopf)
            force    = a + XI_attn^T p_attn + XI_hopf^T a_hopf
            dv       = (-v + force) / tau_v
        """
        v = x[:D]  # (D,)
        h_attn = x[D : D + L]  # (L,)
        h_hopf = x[D + L : D + L + M]  # (M,)

        # hidden targets
        pre_attn = XI_attn @ v + b_attn  # (L,)
        pre_hopf = XI_hopf @ v + c  # (M,)

        # relax hidden states toward their targets
        dh_attn = (pre_attn - h_attn) / tau_h
        dh_hopf = (pre_hopf - h_hopf) / tau_h

        # nonlinear hidden activations
        p_attn = jax.nn.softmax(beta * h_attn)  # (L,)
        a_hopf_act = jnp.maximum(h_hopf, 0.0)  # (M,)

        # visible force
        force = a + XI_attn.T @ p_attn + XI_hopf.T @ a_hopf_act  # (D,)
        dv = (-v + force) / tau_v  # (D,)

        return jnp.concatenate([dv, dh_attn, dh_hopf])

    return rhs_closure


def integrate_dynamics(
    P: DynParams, x0: jnp.ndarray, t1: float, saveat: diffrax.SaveAt
):
    """
    Single, shared diffrax call used everywhere.
    Tsit5 + adaptive steps (PID controller).
    """
    rhs = make_rhs(P)
    ode_term = diffrax.ODETerm(rhs)

    sol = diffrax.diffeqsolve(
        ode_term,
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=float(t1),
        dt0=float(ALPHA),
        y0=x0,
        saveat=saveat,
        # stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=1_000_000,
    )
    return sol


def build_initial_state(
    D: int,
    Xi: jnp.ndarray,
    b_attn: jnp.ndarray,
    XI_hopf: jnp.ndarray,
    c: jnp.ndarray,
    v0: Optional[jnp.ndarray] = None,
):
    """
    Initialize x0 = [v0, h_attn0, h_hopf0].
    We initialize hidden states at their target preactivations.
    """
    if v0 is None:
        v0 = jnp.zeros((D,), dtype=jnp.float32)

    h_attn0 = Xi @ v0 + b_attn  # (L,)
    h_hopf0 = XI_hopf @ v0 + c  # (M,)

    x0 = jnp.concatenate([v0, h_attn0, h_hopf0])
    return x0


def decode_final(v_T: jnp.ndarray, W_out: jnp.ndarray, b_out: jnp.ndarray):
    """
    Same readout head as training.
    If C==1 => binary logistic head.
    Else    => multiclass argmax.
    """
    logits = W_out @ v_T + b_out  # (C,)
    C = logits.shape[0]
    if C == 1:
        prob = jax.nn.sigmoid(logits[0])
        pred = (prob > 0.5).astype(jnp.int32)
        return logits, prob, pred
    else:
        pred = jnp.argmax(logits)
        return logits, None, pred


def infer_single_diffrax(
    ctx_tokens: jnp.ndarray,
    XI_attn: jnp.ndarray,
    XI_hopf: jnp.ndarray,
    a: jnp.ndarray,
    c: jnp.ndarray,
    W_out: jnp.ndarray,
    b_out: jnp.ndarray,
    b_att_template: Optional[jnp.ndarray],
):
    """
    Run inference for one context using Tsit5 + adaptive steps.
    This is reusable and used both for debugging and batched eval.
    """
    Xi_dbg = XI_attn[ctx_tokens]  # (L, D)
    L, D = Xi_dbg.shape

    if b_att_template is None:
        b_attn = jnp.zeros((L,), dtype=jnp.float32)
    else:
        b_attn = b_att_template[:L]

    P = DynParams(
        XI_attn=Xi_dbg,
        XI_hopf=XI_hopf,
        a=a,
        b_attn=b_attn,
        c=c,
        beta=BETA,
        tau_v=TAU_V,
        tau_h=TAU_H,
    )

    x0 = build_initial_state(D, Xi_dbg, b_attn, XI_hopf, c)

    sol = integrate_dynamics(
        P,
        x0,
        t1=T_FINAL,
        saveat=diffrax.SaveAt(t1=True),
    )

    # With SaveAt(t1=True), sol.ys has shape (1, state_dim)
    assert sol.ys is not None, "sol.ys is None"
    x_T = sol.ys.reshape(-1)  # (state_dim,)
    v_T = x_T[:D]  # (D,)

    logits, _, pred = decode_final(v_T, W_out, b_out)
    return pred, logits, v_T


def run_model_direct_inference(ctx_tokens: jnp.ndarray, params: dict):
    """
    Run inference using model_direct's forward-Euler unroll on the given context.
    """
    L = int(Config.L)
    D = int(Config.D)
    if ctx_tokens.ndim != 1 or ctx_tokens.shape[0] != L:
        raise ValueError(f"ctx_tokens must have shape ({L},), got {ctx_tokens.shape}")

    ctx_bits = jnp.asarray(ctx_tokens, dtype=jnp.int32).reshape(1, L)
    V0 = jnp.zeros((1, D), dtype=jnp.float32)

    V_T_batched, _ = infer_forward_euler(params, V0, ctx_bits)  # (1, D)
    logits_batched = logits_from_v(params, V_T_batched)  # (1, C)

    logits = logits_batched[0]
    pred = int(jnp.argmax(logits))
    V_T = V_T_batched[0]
    return logits, pred, V_T


def evaluate_model():
    """
    Evaluate the dynamics model by sampling random bit strings and
    checking classification accuracy against parity labels.
    """
    p = load_params("data/model_best.npz")

    W_enc = p["W_enc"]
    XI_attn = jnp.square(W_enc)

    W_hopf = p["W_hopf"]
    XI_hopf = jnp.square(W_hopf)

    a = p["a"]
    c = p["c"]
    W_dec = p["W_dec"]
    b_dec = p["b_dec"]

    raw_b_attn = p.get("b_attn", 0.0)
    if isinstance(raw_b_attn, float):
        b_att_template = None
    else:
        b_att_template = jnp.asarray(raw_b_attn, dtype=jnp.float32)

    # Debug: single context trajectory
    debug_ctx = jnp.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=jnp.int32)

    Xi_dbg = XI_attn[debug_ctx]
    L_dbg, D_dbg = Xi_dbg.shape

    if b_att_template is None:
        b_att_dbg = jnp.zeros((L_dbg,), dtype=jnp.float32)
    else:
        b_att_dbg = b_att_template[:L_dbg]

    P_dbg = DynParams(
        XI_attn=Xi_dbg,
        XI_hopf=XI_hopf,
        a=a,
        b_attn=b_att_dbg,
        c=c,
        beta=BETA,
        tau_v=TAU_V,
        tau_h=TAU_H,
    )

    x0_dbg = build_initial_state(D_dbg, Xi_dbg, b_att_dbg, XI_hopf, c)

    # reference grid for inspection
    n_ref = max(1, int(T_FINAL / ALPHA))
    ts = jnp.linspace(0.0, T_FINAL, n_ref + 1)

    sol_dbg = integrate_dynamics(
        P_dbg,
        x0_dbg,
        t1=T_FINAL,
        saveat=diffrax.SaveAt(ts=ts),
    )
    assert sol_dbg.ys is not None, "sol_dbg.ys is None"
    ys_dbg = sol_dbg.ys  # (n_ref+1, state_dim)
    v_T_dbg = ys_dbg[-1, :D_dbg]
    logits_dbg, prob_dbg, pred_dbg = decode_final(v_T_dbg, W_dec, b_dec)

    # Compare with forward-Euler training dynamics on the same debug context
    md_logits, md_pred, md_vT = run_model_direct_inference(
        debug_ctx,
        {
            "W_enc": W_enc,
            "W_hopf": W_hopf,
            "a": a,
            "c": c,
            "W_dec": W_dec,
            "b_dec": b_dec,
            "b_attn": (
                jnp.asarray(raw_b_attn) if "b_attn" in p else jnp.zeros((Config.L,))
            ),
        },
    )

    print("=== debug: diffrax dynamics ===")
    print("final_logits:", jnp.asarray(logits_dbg))
    if prob_dbg is not None:
        print("final_prob:", float(prob_dbg))
    print("final_pred:", int(pred_dbg))

    print("=== debug: model_direct ===")
    print("md_logits:", jnp.asarray(md_logits))
    print("md_pred:", md_pred)
    print("||v_T - md_vT||:", float(jnp.linalg.norm(v_T_dbg - md_vT)))

    _, _, test_X, test_y = load_dataset(filename_prefix="parity_data")

    # Forward-Euler logits (reference)
    logits_euler = jnp.array(
        [
            run_model_direct_inference(
                ctx_i,
                {
                    "W_enc": W_enc,
                    "W_hopf": W_hopf,
                    "a": a,
                    "c": c,
                    "W_dec": W_dec,
                    "b_dec": b_dec,
                    "b_attn": (
                        jnp.asarray(raw_b_attn)
                        if "b_attn" in p
                        else jnp.zeros((Config.L,))
                    ),
                },
            )[0]
            for ctx_i in tqdm(test_X)
        ]
    )

    # ------------------------------------------
    # Diffrax-based inference on test set
    # ------------------------------------------
    ctx_bits = jnp.asarray(test_X, dtype=jnp.int32)
    labels = jnp.asarray(test_y, dtype=jnp.int32)

    def vmapped_infer(ctx_tokens):
        pred, logits, _ = infer_single_diffrax(
            ctx_tokens,
            XI_attn,
            XI_hopf,
            a,
            c,
            W_dec,
            b_dec,
            b_att_template,
        )
        return pred, logits

    preds, logits_diffrax = jax.vmap(vmapped_infer)(ctx_bits)
    acc = jnp.mean((preds == labels).astype(jnp.float32))

    plt.scatter(logits_diffrax[:, 0], logits_euler[:, 0])
    plt.xlabel("logits from diffrax solver")
    plt.ylabel("logits from manual forward Euler")
    plt.show()

    print(f"Eval: {len(logits_diffrax)} samples | accuracy = {float(acc):.3f}")
    return float(acc)


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
