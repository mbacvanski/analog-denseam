import functools
import math
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.tree_util import DictKey

from config import Config
from utils import ModelParams


def init_params(key: jax.Array) -> ModelParams:
    k1, k2, k3 = jr.split(key, 3)

    # token embedding (square to get xi_attn_embed)
    xi_attn_embed_raw = (
        jr.normal(k1, (Config.vocab_size, Config.D)) * Config.xi_attn_embed_raw_scale
    )

    # Hopfield memory (square to get xi_hopf)
    xi_hopf_raw = jr.normal(k2, (Config.M, Config.D)) * Config.xi_hopf_raw_scale

    # attention bias
    b = jnp.zeros((Config.L,))

    # Hopfield bias
    c = jnp.zeros((Config.M,))

    # visible bias
    a = jnp.zeros((Config.D,))

    # decoder weights and bias for parity classification
    W_dec = jr.normal(k3, (Config.vocab_size, Config.D)) * (1.0 / jnp.sqrt(Config.D))
    b_dec = jnp.zeros((Config.vocab_size,))

    return dict(
        xi_attn_embed_raw=xi_attn_embed_raw,
        xi_hopf_raw=xi_hopf_raw,
        b=b,
        c=c,
        a=a,
        W_dec=W_dec,
        b_dec=b_dec,
    )


def get_xi_attn_embed(params: ModelParams) -> jax.Array:
    """ensure positive values for xi_attn"""
    return jnp.square(params["xi_attn_embed_raw"])


def get_xi_hopf(params: ModelParams) -> jax.Array:
    """ensure positive values for xi_hopf"""
    return jnp.square(params["xi_hopf_raw"])


def L_attn(h: jax.Array) -> jax.Array:  # (B,L) -> (B,)
    return (1.0 / Config.beta) * jax.nn.logsumexp(Config.beta * h, axis=-1)


def L_hopf(h: jax.Array) -> jax.Array:  # (B,M) -> (B,)
    r = jnp.maximum(h, 0.0)
    return 0.5 * jnp.sum(r * r, axis=-1)


# ------------------------------------------------------------
# Mixed-coordinate energy
# ------------------------------------------------------------
def energy_per_sample(params, v, h_attn, h_hopf, p, a, ctx_bits_row):
    # Xi: (L, D) rows selected by the tokens in this sequence (positive-used)
    xi_attn_embed = get_xi_attn_embed(params)  # (vocab_size, D)
    xi_attn = xi_attn_embed[ctx_bits_row]  # (L, D)

    # Visible quadratic
    dv = v - params["a"]
    vis_term = 0.5 * jnp.dot(dv, dv)  # scalar

    # Couplings (positive-used eta)
    coupling = jnp.dot(v, xi_attn.T @ p + get_xi_hopf(params).T @ a)  # scalar

    # Linear Fenchel-Young-style saddle terms (activation coords)
    attn_bias = jnp.dot(p, h_attn - params["b"])  # scalar
    hopf_bias = jnp.dot(a, h_hopf - params["c"])  # scalar

    # Total mixed-coordinate energy
    return vis_term - coupling + attn_bias + hopf_bias - L_attn(h_attn) - L_hopf(h_hopf)


def energy_per_batch(params, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits):
    """
    V:      (B, D)
    H_att:  (B, L)
    H_hopf: (B, M)
    P:      (B, L)   -- activations (softmax of H_att), treated as independent arg
    A:      (B, M)   -- activations (ReLU of H_hopf), treated as independent arg
    ctx_bits: (B, L)
    """
    E_b = jax.vmap(energy_per_sample, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits
    )
    return jnp.sum(E_b)


# ------------------------------------------------------------
# Inference: forward Euler on activation-gradient flow
# ------------------------------------------------------------
def _init_hidden(params, V0, ctx_bits):
    """
    Initialize hidden states at their targets for V0.
    H_attn0 = Xi V0 + b
    H_hopf0 = eta V0 + c
    """
    # Xi_seq: (B, L, D) using positive-used xi
    xi_attn_embed = get_xi_attn_embed(params)  # (vocab_size, D)
    batch_xi_attn = xi_attn_embed[ctx_bits]  # (B, L, D)
    H_attn0 = jnp.einsum("bld,bd->bl", batch_xi_attn, V0) + params["b"]  # (B, L)

    # Positive-used eta
    H_hopf0 = V0 @ get_xi_hopf(params).T + params["c"]  # (B, M)
    return H_attn0, H_hopf0


@functools.partial(jax.jit, donate_argnums=(1,))  # donate V0 buffer
def infer_forward_euler_with_force(
    params: ModelParams, V0: jax.Array, ctx_bits: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Returns:
      V_T: (B, D) terminal visible state
      F_T: (B, D) force at V_T, i.e. dV/dt = -(1/tau_v) * dE/dV at V_T
    """
    xi_attn_embed = get_xi_attn_embed(params)  # (vocab_size, D)
    batch_xi_attn = xi_attn_embed[ctx_bits]  # (B, L, D)
    step_v = Config.step_size / Config.tau_v
    step_h = Config.step_size / Config.tau_h

    # Initial pre-activations
    H_attn0 = jnp.einsum("bld,bd->bl", batch_xi_attn, V0) + params["b"]
    H_hopf0 = V0 @ get_xi_hopf(params).T + params["c"]

    # Energy for batch with Xi captured (no per-step allocations beyond carry)
    def batch_energy(params, V, H_attn, H_hopf, F_attn, F_hopf):
        def _sample_energy(v, h_attn, h_hopf, f_attn, f_hopf, xi_attn):
            dv = v - params["a"]
            vis = 0.5 * jnp.dot(dv, dv)
            coupling = jnp.dot(v, xi_attn.T @ f_attn + get_xi_hopf(params).T @ f_hopf)
            att_bias = jnp.dot(f_attn, h_attn - params["b"])
            hopf_bias = jnp.dot(f_hopf, h_hopf - params["c"])
            return (
                vis - coupling + att_bias + hopf_bias - L_attn(h_attn) - L_hopf(h_hopf)
            )

        Eb = jax.vmap(_sample_energy, in_axes=(0, 0, 0, 0, 0, 0))(
            V, H_attn, H_hopf, F_attn, F_hopf, batch_xi_attn
        )
        return jnp.sum(Eb)

    grad_E = jax.grad(batch_energy, argnums=(1, 4, 5))  # grads wrt (V, F_attn, F_hopf)

    def grads_activation(V, H_attn, H_hopf):
        F_attn = jax.nn.softmax(Config.beta * H_attn, axis=-1)
        F_hopf = jnp.maximum(H_hopf, 0.0)
        dE_dV, dE_dF_attn, dE_dF_hopf = grad_E(
            params, V, H_attn, H_hopf, F_attn, F_hopf
        )
        return dE_dV, dE_dF_attn, dE_dF_hopf

    def body(_, carry):
        V, H_attn, H_hopf = carry
        dE_dV, dE_dF_attn, dE_dF_hopf = grads_activation(V, H_attn, H_hopf)
        V = V - step_v * dE_dV
        H_attn = H_attn - step_h * dE_dF_attn
        H_hopf = H_hopf - step_h * dE_dF_hopf
        return (V, H_attn, H_hopf)

    V_T, H_attn_T, H_hopf_T = jax.lax.fori_loop(
        0, Config.n_steps, body, (V0, H_attn0, H_hopf0)
    )

    # Force at terminal state (no extra allocations beyond one grad eval)
    dE_dV_T, _, _ = grads_activation(V_T, H_attn_T, H_hopf_T)
    F_T = -(1.0 / Config.tau_v) * dE_dV_T
    return V_T, F_T


@jax.jit
def infer_forward_euler(params, V0, ctx_bits):
    """
    Returns:
      V_T: (B, D)
      traj: dict of optional trajectories (currently V only)
    """
    xi_attn_embed = get_xi_attn_embed(params)  # (vocab_size, D)
    batch_xi_attn = xi_attn_embed[ctx_bits]  # (B, L, D)
    H_attn0 = jnp.einsum("bld,bd->bl", batch_xi_attn, V0) + params["b"]
    H_hopf0 = V0 @ get_xi_hopf(params).T + params["c"]

    # Pack everything needed for grads to avoid repeated closures
    def energy_batch_w_Xi(params, V, H_attn, H_hopf, F_attn, F_hopf):
        # same as energy_per_batch, but use Xi_seq captured from outer scope
        def energy_per_sample_wXi(v, h_attn, h_hopf, f_attn, f_hopf, xi_attn):
            dv = v - params["a"]
            vis = 0.5 * jnp.dot(dv, dv)
            coupling = jnp.dot(v, xi_attn.T @ f_attn + get_xi_hopf(params).T @ f_hopf)
            att_bias = jnp.dot(f_attn, h_attn - params["b"])
            hopf_bias = jnp.dot(f_hopf, h_hopf - params["c"])
            return (
                vis - coupling + att_bias + hopf_bias - L_attn(h_attn) - L_hopf(h_hopf)
            )

        Eb = jax.vmap(energy_per_sample_wXi, in_axes=(0, 0, 0, 0, 0, 0))(
            V, H_attn, H_hopf, F_attn, F_hopf, batch_xi_attn
        )
        return jnp.sum(Eb)

    grad_E = jax.grad(
        energy_batch_w_Xi, argnums=(1, 4, 5)
    )  # grads w.r.t. (V, F_attn, F_hopf)

    def grads_activation(V, H_attn, H_hopf):
        F_attn = jax.nn.softmax(Config.beta * H_attn, axis=-1)
        F_hopf = jnp.maximum(H_hopf, 0.0)
        (
            dE_dV,
            dE_dF_attn,
            dE_dF_hopf,
        ) = grad_E(params, V, H_attn, H_hopf, F_attn, F_hopf)
        return dE_dV, dE_dF_attn, dE_dF_hopf, F_attn, F_hopf

    def step(carry, _):
        V, H_attn, H_hopf = carry
        dE_dV, dE_dF_attn, dE_dF_hopf, _, _ = grads_activation(V, H_attn, H_hopf)
        Vn = V - (Config.step_size / Config.tau_v) * dE_dV
        H_attn_n = H_attn - (Config.step_size / Config.tau_h) * dE_dF_attn
        H_hopf_n = H_hopf - (Config.step_size / Config.tau_h) * dE_dF_hopf
        return (Vn, H_attn_n, H_hopf_n), Vn

    (V_T, _, _), V_traj = jax.lax.scan(
        step, (V0, H_attn0, H_hopf0), xs=None, length=Config.n_steps
    )
    return V_T, V_traj


# ------------------------------------------------------------
# Direct inference with full trajectories
# ------------------------------------------------------------
def run_model_inference_steps(ctx_bits, params, V0=None):
    """
    Run forward Euler inference and return trajectories of
    logits, visible V, hidden preactivations (H_att, H_hopf),
    and per-sample mixed energy for all steps.

    Shapes:
      ctx_bits: (B, L)
      V0 (optional): (B, D), defaults to zeros
    """
    B = ctx_bits.shape[0]
    V0 = jnp.zeros((B, Config.D), dtype=jnp.float32)

    H_attn0, H_hopf0 = _init_hidden(params, V0, ctx_bits)

    def step(carry, _):
        V, H_attn, H_hopf = carry

        # Activations at current state (used only for gradients)
        F_attn = jax.nn.softmax(Config.beta * H_attn, axis=-1)  # (B, L)
        F_hopf = jnp.maximum(H_hopf, 0.0)  # (B, M)

        # Gradients wrt activation-coordinates
        dE_dV = jax.grad(energy_per_batch, argnums=1)(
            params, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits
        )
        dE_dF_attn = jax.grad(energy_per_batch, argnums=4)(
            params, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits
        )
        dE_dF_hopf = jax.grad(energy_per_batch, argnums=5)(
            params, V, H_attn, H_hopf, F_attn, F_hopf, ctx_bits
        )

        # Forward Euler updates
        Vn = V - (Config.step_size / Config.tau_v) * dE_dV
        H_attn_n = H_attn - (Config.step_size / Config.tau_h) * dE_dF_attn
        H_hopf_n = H_hopf - (Config.step_size / Config.tau_h) * dE_dF_hopf

        # Log post-update quantities (time t+Δt)
        F_attn_n = jax.nn.softmax(Config.beta * H_attn_n, axis=-1)  # (B, L)
        H_hopf_n = jnp.maximum(H_hopf_n, 0.0)  # (B, M)
        logits_n = logits_from_v(params, Vn)  # (B, vocab_size)

        # Per-sample mixed energy at post-update state
        E_b = jax.vmap(energy_per_sample, in_axes=(None, 0, 0, 0, 0, 0, 0))(
            params, Vn, H_attn_n, H_hopf_n, F_attn_n, H_hopf_n, ctx_bits
        )  # (B,)

        return (Vn, H_attn_n, H_hopf_n), (Vn, H_attn_n, H_hopf_n, logits_n, E_b)

    # Scan for T steps; collect trajectories
    (V_T, H_attn_T, H_hopf_T), (
        V_traj,
        H_attn_traj,
        H_hopf_traj,
        logits_traj,
        E_traj,
    ) = jax.lax.scan(step, (V0, H_attn0, H_hopf0), xs=None, length=Config.n_steps)

    traj = dict(
        V=V_traj,  # (T, B, D)
        H_att=H_attn_traj,  # (T, B, L)
        H_hopf=H_hopf_traj,  # (T, B, M)
        logits=logits_traj,  # (T, B, vocab_size)
        energy=E_traj,  # (T, B)  per-sample energies
    )
    return (V_T, H_attn_T, H_hopf_T), traj


# ------------------------------------------------------------
# Readout and loss
# ------------------------------------------------------------
def logits_from_v(params: ModelParams, V: jax.Array) -> jax.Array:
    return V @ params["W_dec"].T + params["b_dec"]  # (B, vocab_size)


def force_penalty(F_T: jax.Array) -> jax.Array:
    return jnp.mean(jnp.sum(F_T * F_T, axis=1))  # MSE of force


@jax.jit
def loss_fn(
    params: ModelParams, ctx_bits: jax.Array, labels: jax.Array, force_weight: jax.Array
):
    """
    force_weight: scalar multiplier for the force penalty term.
    """
    B = ctx_bits.shape[0]
    V0 = jnp.zeros((B, Config.D))  # visible init
    V_T, F_T = infer_forward_euler_with_force(params, V0, ctx_bits)
    logits_T = logits_from_v(params, V_T)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits_T, labels).mean()
    return ce + force_weight * force_penalty(F_T)


@jax.jit
def evaluate(params: ModelParams, valid_X: jax.Array, valid_y: jax.Array) -> jax.Array:
    V_T, _ = infer_forward_euler_with_force(
        params, jnp.zeros((valid_y.shape[0], Config.D), jnp.float32), valid_X
    )
    preds = jnp.argmax(logits_from_v(params, V_T), axis=1)
    return jnp.mean((preds == valid_y).astype(jnp.float32))


def force_penalty_weight(epoch: int) -> float:
    """
    Around 17k epochs the model starts to get 100% accuracy on the validation set,
    at which point start ramping up the force penalty.
    Applying force penalty too early causes the model to never reach good accuracies.
    """

    start = Config.force_penalty_start
    duration = Config.force_penalty_duration
    end = start + duration

    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0

    # normalized progress in [0,1]
    t = (epoch - start) / duration

    # cosine ramp: 0 → 1
    return Config.force_penalty_scale * (1.0 - math.cos(math.pi * t))


# Parameter partition
def label_tree(params: ModelParams):
    def label_from_path(path, leaf) -> str:
        last = path[-1]
        if isinstance(last, DictKey):
            name = last.key  # 'xi_attn_embed_raw', 'xi_hopf_raw', etc.
        else:
            name = str(last)
        if name in ("xi_attn_embed_raw", "xi_hopf_raw"):
            return "fast"
        else:
            return "slow"

    return jax.tree_util.tree_map_with_path(label_from_path, params)
