"""
Perform inference on the parity dataset using the pre-trained weights from model_energy_train.py,
but using an explicit ODE solver (diffrax) to integrate the dynamics instead of the unrolled forward-Euler
used during training.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pickle
from dataclasses import dataclass
from typing import Optional

import diffrax
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from tqdm import tqdm

import model_energy_train as md

# ----------------------------
# Config
# ----------------------------
PICKLE_PATH = "model_best.pkl"
BETA = md.Cfg.beta
ALPHA = md.Cfg.step_size / 10  # initial dt guess for the ODE solver
T_FINAL = float(md.Cfg.T_final)
TAU_V = md.Cfg.tau_v  # visible neuron time constant (arbitrary units)
TAU_H = md.Cfg.tau_h  # hidden neuron time constant (faster relax)


# ---------------------------------
# Dynamics params container
# ---------------------------------
@dataclass
class DynParams:
    Xi: jnp.ndarray    # (L_ctx, D)
    eta: jnp.ndarray   # (M, D)
    a: jnp.ndarray     # (D,)
    b_att: jnp.ndarray # (L_ctx,)
    c: jnp.ndarray     # (M,)
    beta: float        # scalar
    tau_v: float       # scalar
    tau_h: float       # scalar


def make_rhs(P: DynParams):
    """
    Return a vector field fn(t, x, args_unused) -> dx/dt
    with P closed over so we don't have to pass P as `args`.
    """
    Xi = P.Xi        # (L_ctx, D)
    eta = P.eta      # (M, D)
    a = P.a          # (D,)
    b_att = P.b_att  # (L_ctx,)
    c = P.c          # (M,)
    beta = P.beta
    tau_v = P.tau_v
    tau_h = P.tau_h

    L_ctx = Xi.shape[0]
    D = Xi.shape[1]
    M = eta.shape[0]

    def rhs_closure(t, x, _):
        """
        State x = concat[v (D), h_att (L_ctx), h_hopf (M)].

        Hidden:
            pre_att  = Xi @ v + b_att
            pre_hopf = eta @ v + c
            dh_att   = (pre_att  - h_att)  / tau_h
            dh_hopf  = (pre_hopf - h_hopf) / tau_h

        Visible:
            p_att    = softmax(beta * h_att)
            a_hopf   = relu(h_hopf)
            force    = a + Xi^T p_att + eta^T a_hopf
            dv       = (-v + force) / tau_v
        """
        v = x[:D]                               # (D,)
        h_att = x[D:D + L_ctx]                  # (L_ctx,)
        h_hopf = x[D + L_ctx:D + L_ctx + M]     # (M,)

        # hidden targets
        pre_att = Xi @ v + b_att                # (L_ctx,)
        pre_hopf = eta @ v + c                  # (M,)

        # relax hidden states toward their targets
        dh_att = (pre_att - h_att) / tau_h
        dh_hopf = (pre_hopf - h_hopf) / tau_h

        # nonlinear hidden activations
        p_att = jax.nn.softmax(beta * h_att)    # (L_ctx,)
        a_hopf_act = jnp.maximum(h_hopf, 0.0)   # (M,)

        # visible force
        force = a + Xi.T @ p_att + eta.T @ a_hopf_act  # (D,)
        dv = (-v + force) / tau_v                      # (D,)

        return jnp.concatenate([dv, dh_att, dh_hopf])

    return rhs_closure


def integrate_dynamics(P: DynParams, x0: jnp.ndarray, t1: float, saveat: diffrax.SaveAt):
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
    b_att: jnp.ndarray,
    eta: jnp.ndarray,
    c: jnp.ndarray,
    v0: Optional[jnp.ndarray] = None,
):
    """
    Initialize x0 = [v0, h_att0, h_hopf0].
    We initialize hidden states at their target preactivations.
    """
    if v0 is None:
        v0 = jnp.zeros((D,), dtype=jnp.float32)

    h_att0 = Xi @ v0 + b_att  # (L_ctx,)
    h_hopf0 = eta @ v0 + c    # (M,)

    x0 = jnp.concatenate([v0, h_att0, h_hopf0])
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
    xi_emb: jnp.ndarray,
    eta: jnp.ndarray,
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
    Xi = xi_emb[ctx_tokens]  # (L_ctx, D)
    L_ctx, D = Xi.shape

    if b_att_template is None:
        b_att = jnp.zeros((L_ctx,), dtype=jnp.float32)
    else:
        b_att = b_att_template[:L_ctx]

    P = DynParams(
        Xi=Xi,
        eta=eta,
        a=a,
        b_att=b_att,
        c=c,
        beta=jnp.asarray(BETA, dtype=jnp.float32),
        tau_v=jnp.asarray(TAU_V, dtype=jnp.float32),
        tau_h=jnp.asarray(TAU_H, dtype=jnp.float32),
    )

    x0 = build_initial_state(D, Xi, b_att, eta, c)

    sol = integrate_dynamics(
        P,
        x0,
        t1=T_FINAL,
        saveat=diffrax.SaveAt(t1=True),
    )

    # With SaveAt(t1=True), sol.ys has shape (1, state_dim)
    x_T = sol.ys.reshape(-1)  # (state_dim,)
    v_T = x_T[:D]             # (D,)

    logits, _, pred = decode_final(v_T, W_out, b_out)
    return pred, logits, v_T


def run_model_direct_inference(ctx_tokens: jnp.ndarray, params: dict):
    """
    Run inference using model_direct's forward-Euler unroll on the given context.
    """
    L = int(md.Cfg.L)
    D = int(md.Cfg.D)
    if ctx_tokens.ndim != 1 or ctx_tokens.shape[0] != L:
        raise ValueError(f"ctx_tokens must have shape ({L},), got {ctx_tokens.shape}")

    ctx_bits = jnp.asarray(ctx_tokens, dtype=jnp.int32).reshape(1, L)
    V0 = jnp.zeros((1, D), dtype=jnp.float32)

    V_T_batched, _ = md.infer_forward_euler(params, V0, ctx_bits)  # (1, D)
    logits_batched = md.logits_from_v(params, V_T_batched)         # (1, C)

    logits = logits_batched[0]
    pred = int(jnp.argmax(logits))
    V_T = V_T_batched[0]
    return logits, pred, V_T


def evaluate_model(n_samples: int = 256, seed: int = 0):
    """
    Evaluate the dynamics model by sampling random bit strings and
    checking classification accuracy against parity labels.
    """
    # ------------------------------------------
    # Load trained parameters
    # ------------------------------------------
    with open(PICKLE_PATH, "rb") as f:
        p = pickle.load(f)

    xi_emb_raw = jnp.asarray(p["xi_emb"], dtype=jnp.float32)
    xi_emb = jnp.square(xi_emb_raw)

    eta_raw = jnp.asarray(p["eta"], dtype=jnp.float32)
    eta = jnp.square(eta_raw)

    a = jnp.asarray(p["a_v"], dtype=jnp.float32)
    c = jnp.asarray(p["c"], dtype=jnp.float32)
    W_out = jnp.asarray(p["W_out"], dtype=jnp.float32)
    b_out = jnp.asarray(p["b_out"], dtype=jnp.float32)

    raw_b_att = p.get("b_att", 0.0)
    if isinstance(raw_b_att, float):
        b_att_template = None
    else:
        b_att_template = jnp.asarray(raw_b_att, dtype=jnp.float32)

    # ------------------------------------------
    # Debug: single context trajectory
    # ------------------------------------------
    debug_ctx = jnp.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=jnp.int32)

    Xi_dbg = xi_emb[debug_ctx]
    L_dbg, D_dbg = Xi_dbg.shape

    if b_att_template is None:
        b_att_dbg = jnp.zeros((L_dbg,), dtype=jnp.float32)
    else:
        b_att_dbg = b_att_template[:L_dbg]

    P_dbg = DynParams(
        Xi=Xi_dbg,
        eta=eta,
        a=a,
        b_att=b_att_dbg,
        c=c,
        beta=BETA,
        tau_v=TAU_V,
        tau_h=TAU_H,
    )

    x0_dbg = build_initial_state(D_dbg, Xi_dbg, b_att_dbg, eta, c)

    # reference grid for inspection
    n_ref = max(1, int(T_FINAL / ALPHA))
    ts = jnp.linspace(0.0, T_FINAL, n_ref + 1)

    sol_dbg = integrate_dynamics(
        P_dbg,
        x0_dbg,
        t1=T_FINAL,
        saveat=diffrax.SaveAt(ts=ts),
    )
    ys_dbg = sol_dbg.ys  # (n_ref+1, state_dim)
    v_T_dbg = ys_dbg[-1, :D_dbg]
    logits_dbg, prob_dbg, pred_dbg = decode_final(v_T_dbg, W_out, b_out)

    # Compare with forward-Euler training dynamics on the same debug context
    md_logits, md_pred, md_vT = run_model_direct_inference(debug_ctx, {
        "xi_emb": xi_emb_raw,
        "eta": eta_raw,
        "a_v": a,
        "c": c,
        "W_out": W_out,
        "b_out": b_out,
        "b_att": jnp.asarray(raw_b_att) if "b_att" in p else jnp.zeros((md.Cfg.L,))
    })

    print("=== debug: diffrax dynamics ===")
    print("final_logits:", jnp.asarray(logits_dbg))
    if prob_dbg is not None:
        print("final_prob:", float(prob_dbg))
    print("final_pred:", int(pred_dbg))

    print("=== debug: model_direct ===")
    print("md_logits:", jnp.asarray(md_logits))
    print("md_pred:", md_pred)
    print("||v_T - md_vT||:", float(jnp.linalg.norm(v_T_dbg - md_vT)))

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------
    from model_energy_train import load_dataset
    _, _, test_X, test_y = load_dataset(filename_prefix="parity_data")

    # Forward-Euler logits (reference)
    logits_euler = jnp.array([
        run_model_direct_inference(ctx_i, {
            "xi_emb": xi_emb_raw,
            "eta": eta_raw,
            "a_v": a,
            "c": c,
            "W_out": W_out,
            "b_out": b_out,
            "b_att": jnp.asarray(raw_b_att) if "b_att" in p else jnp.zeros((md.Cfg.L,))
        })[0]
        for ctx_i in tqdm(test_X)
    ])

    # ------------------------------------------
    # Diffrax-based inference on test set
    # ------------------------------------------
    ctx_bits = jnp.asarray(test_X, dtype=jnp.int32)
    labels = jnp.asarray(test_y, dtype=jnp.int32)

    def vmapped_infer(ctx_tokens):
        pred, logits, _ = infer_single_diffrax(
            ctx_tokens,
            xi_emb,
            eta,
            a,
            c,
            W_out,
            b_out,
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
    evaluate_model(n_samples=1000, seed=0)


if __name__ == "__main__":
    main()
