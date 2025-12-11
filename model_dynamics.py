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
    xi_attn: jnp.ndarray  # (L, D)
    xi_hopf: jnp.ndarray  # (M, D)
    a: jnp.ndarray  # (D,)
    b: jnp.ndarray  # (L,)
    c: jnp.ndarray  # (M,)
    beta: float  # scalar
    tau_v: float  # scalar
    tau_h: float  # scalar


def make_rhs(P: DynParams):
    """
    Return a vector field fn(t, x, args_unused) -> dx/dt
    with P closed over so we don't have to pass P as `args`.
    """
    xi_attn = P.xi_attn  # (L, D)
    xi_hopf = P.xi_hopf  # (M, D)
    a = P.a  # (D,)
    b = P.b  # (L,)
    c = P.c  # (M,)
    beta = P.beta
    tau_v = P.tau_v
    tau_h = P.tau_h

    L = xi_attn.shape[0]
    D = xi_attn.shape[1]
    M = xi_hopf.shape[0]

    def rhs_closure(t, x, _):
        """
        State x = concat[v (D), h_attn (L), h_hopf (M)].

        Hidden:
            pre_attn = xi_attn @ v + b
            pre_hopf = xi_hopf @ v + c
            dh_attn  = (pre_attn  - h_attn)  / tau_h
            dh_hopf  = (pre_hopf - h_hopf) / tau_h

        Visible:
            f_attn   = softmax(beta * h_attn)
            f_hopf   = relu(h_hopf)
            force    = a + xi_attn^T f_attn + xi_hopf^T a_hopf
            dv       = (-v + force) / tau_v
        """
        v = x[:D]  # (D,)
        h_attn = x[D : D + L]  # (L,)
        h_hopf = x[D + L : D + L + M]  # (M,)

        # hidden targets
        pre_attn = xi_attn @ v + b  # (L,)
        pre_hopf = xi_hopf @ v + c  # (M,)

        # relax hidden states toward their targets
        dh_attn = (pre_attn - h_attn) / tau_h
        dh_hopf = (pre_hopf - h_hopf) / tau_h

        # nonlinear hidden activations
        f_attn = jax.nn.softmax(beta * h_attn)  # (L,)
        f_hopf = jnp.maximum(h_hopf, 0.0)  # (M,)

        # force on visible neurons
        force = a + xi_attn.T @ f_attn + xi_hopf.T @ f_hopf  # (D,)
        dv = (-v + force) / tau_v  # (D,)

        return jnp.concatenate([dv, dh_attn, dh_hopf])

    return rhs_closure


def integrate_dynamics(
    P: DynParams, x_0: jnp.ndarray, t1: float, saveat: diffrax.SaveAt
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
        y0=x_0,
        saveat=saveat,
        # stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=1_000_000,
    )
    return sol


def build_initial_state(
    D: int,
    xi_attn: jnp.ndarray,
    xi_hopf: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    v_0: Optional[jnp.ndarray] = None,
):
    """
    Initialize x_0 = [v0, h_attn_0, h_hopf_0].
    We initialize hidden states at their target preactivations.
    """
    if v_0 is None:
        v_0 = jnp.zeros((D,), dtype=jnp.float32)

    h_attn_0 = xi_attn @ v_0 + b  # (L,)
    h_hopf_0 = xi_hopf @ v_0 + c  # (M,)

    x_0 = jnp.concatenate([v_0, h_attn_0, h_hopf_0])
    return x_0


def decode_final(v_T: jnp.ndarray, W_out: jnp.ndarray, b_out: jnp.ndarray):
    """
    Same readout head as training.
    If n_categories==1 => binary logistic head.
    Else    => multiclass argmax.
    """
    logits = W_out @ v_T + b_out  # (n_categories,)
    n_categories = logits.shape[0]
    if n_categories == 1:
        prob = jax.nn.sigmoid(logits[0])
        pred = (prob > 0.5).astype(jnp.int32)
        return logits, prob, pred
    else:
        pred = jnp.argmax(logits)
        return logits, None, pred


def infer_single_diffrax(
    ctx_tokens: jnp.ndarray,
    xi_attn_emb: jnp.ndarray,
    xi_hopf: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    w_out: jnp.ndarray,
    b_out: jnp.ndarray,
):
    """
    Run inference for one context using Tsit5 + adaptive steps.
    This is reusable and used both for debugging and batched eval.
    """
    xi_attn = xi_attn_emb[ctx_tokens]  # (L, D)
    L, D = xi_attn.shape

    P = DynParams(
        xi_attn=xi_attn,
        xi_hopf=xi_hopf,
        a=a,
        b=b,
        c=c,
        beta=BETA,
        tau_v=TAU_V,
        tau_h=TAU_H,
    )

    x_0 = build_initial_state(D, xi_attn, xi_hopf, b, c)

    sol = integrate_dynamics(P, x_0, t1=T_FINAL, saveat=diffrax.SaveAt(t1=True))

    # With SaveAt(t1=True), sol.ys has shape (1, state_dim)
    assert sol.ys is not None, "sol.ys is None"
    x_T = sol.ys.reshape(-1)  # (state_dim,)
    v_T = x_T[:D]  # (D,)

    logits, _, pred = decode_final(v_T, w_out, b_out)
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
    v_0 = jnp.zeros((1, D), dtype=jnp.float32)

    V_T_batched, _ = infer_forward_euler(params, v_0, ctx_bits)  # (1, D)
    logits_batched = logits_from_v(params, V_T_batched)  # (1, C)

    logits = logits_batched[0]
    pred = int(jnp.argmax(logits))
    v_T = V_T_batched[0]
    return logits, pred, v_T


def evaluate_model():
    """
    Evaluate the dynamics model by sampling random bit strings and
    checking classification accuracy against parity labels.
    """
    # Debug: single context trajectory
    debug_ctx = jnp.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=jnp.int32)

    P = load_params("data/model.npz")

    xi_attn_emb_raw = P["xi_attn_emb"]
    xi_attn_emb = jnp.square(xi_attn_emb_raw)
    xi_attn = xi_attn_emb[debug_ctx]
    L_dbg, D_dbg = xi_attn.shape

    xi_hopf_raw = P["xi_hopf"]
    xi_hopf = jnp.square(xi_hopf_raw)

    a = P["a"]
    c = P["c"]
    w_dec = P["w_dec"]
    b_dec = P["b_dec"]

    b = P.get("b", 0.0)
    if isinstance(b, float):
        b_att_dbg = jnp.zeros((L_dbg,), dtype=jnp.float32)
    else:
        b_att_dbg = b[:L_dbg]

    P_dbg = DynParams(
        xi_attn=xi_attn,
        xi_hopf=xi_hopf,
        a=a,
        b=b_att_dbg,
        c=c,
        beta=BETA,
        tau_v=TAU_V,
        tau_h=TAU_H,
    )

    x0_dbg = build_initial_state(D_dbg, xi_attn, xi_hopf, b_att_dbg, c)

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
    logits_dbg, prob_dbg, pred_dbg = decode_final(v_T_dbg, w_dec, b_dec)

    # Compare with forward-Euler training dynamics on the same debug context
    md_logits, md_pred, md_vT = run_model_direct_inference(
        debug_ctx,
        {
            "xi_attn_emb": xi_attn_emb_raw,
            "xi_hopf": xi_hopf_raw,
            "a": a,
            "c": c,
            "w_dec": w_dec,
            "b_dec": b_dec,
            "b": (
                jnp.asarray(b) if "b" in P else jnp.zeros((Config.L,))
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
                    "xi_attn_emb": xi_attn_emb_raw,
                    "xi_hopf": xi_hopf_raw,
                    "a": a,
                    "b": b,
                    "c": c,
                    "w_dec": w_dec,
                    "b_dec": b_dec,
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
            xi_attn_emb,
            xi_hopf,
            a,
            b,
            c,
            w_dec,
            b_dec,
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
