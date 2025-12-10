import functools
import math

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.tree_util import DictKey

from config import Config
from utils import ModelParams


def init_params(key: jax.Array) -> ModelParams:
    k1, k2, k3 = jr.split(key, 3)

    # token embedding (square to get XI_attn)
    xi_attn_emb = jr.normal(k1, (Config.vocab_size, Config.D)) * Config.xi_attn_emb_scale

    # Hopfield memory (square to get XI_hopf)
    xi_hopf = jr.normal(k2, (Config.M, Config.D)) * Config.xi_hopf_scale

    # visible bias
    a = jnp.zeros((Config.D,))

    # attention bias
    b = jnp.zeros((Config.L,))

    # Hopfield bias
    c = jnp.zeros((Config.M,))
    
    # decoder weights and bias for parity classification
    w_dec = jr.normal(k3, (Config.vocab_size, Config.D)) * (1.0 / jnp.sqrt(Config.D))
    b_dec = jnp.zeros((Config.vocab_size,))

    return dict(
        xi_attn_emb=xi_attn_emb,
        xi_hopf=xi_hopf,
        a=a,
        b=b,
        c=c,
        w_dec=w_dec,
        b_dec=b_dec,
    )


def get_xi_attn_emb(params: ModelParams) -> jax.Array:
    """ensure positive values for XI_attn"""
    return jnp.square(params["xi_attn_emb"])


def get_xi_hopf(params: ModelParams) -> jax.Array:
    """ensure positive values for XI_hopf"""
    return jnp.square(params["xi_hopf"])


def L_attn(h: jax.Array) -> jax.Array:  # (B,L) -> (B,)
    return (1.0 / Config.beta) * jax.nn.logsumexp(Config.beta * h, axis=-1)


def L_hopf(h: jax.Array) -> jax.Array:  # (B,M) -> (B,)
    r = jnp.maximum(h, 0.0)
    return 0.5 * jnp.sum(r * r, axis=-1)


# ------------------------------------------------------------
# Mixed-coordinate energy
# ------------------------------------------------------------
def energy_per_sample(params, v, h_att, h_hopf, p, a, ctx_bits_row):
    # xi_attn: (L, D) rows selected by the tokens in this sequence (positive-used)
    xi_attn_emb = get_xi_attn_emb(params)  # (vocab_size, D)
    xi_attn = xi_attn_emb[ctx_bits_row]  # (L, D)

    # Visible quadratic
    dv = v - params["a"]
    vis_term = 0.5 * jnp.dot(dv, dv)  # scalar

    # Couplings (positive-used eta)
    XiTp = xi_attn.T @ p  # (D,)
    etaTa = get_xi_hopf(params).T @ a  # (D,)
    coupling = jnp.dot(v, XiTp + etaTa)  # scalar

    # Linear Fenchel-Young-style saddle terms (activation coords)
    attn_bias = jnp.dot(p, h_att - params["b"])  # scalar
    hopf_bias = jnp.dot(a, h_hopf - params["c"])  # scalar

    # Total mixed-coordinate energy
    return vis_term - coupling + attn_bias + hopf_bias - L_attn(h_att) - L_hopf(h_hopf)


def energy_per_batch(params, V, h_attn, h_hopf, f_attn, f_hopf, ctx_bits):
    """
    V:      (B, D)
    h_attn:  (B, L)
    h_hopf: (B, M)
    P:      (B, L)   -- activations (softmax of h_attn), treated as independent arg
    A:      (B, M)   -- activations (ReLU of h_hopf), treated as independent arg
    ctx_bits: (B, L)
    """
    E_b = jax.vmap(energy_per_sample, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, V, h_attn, h_hopf, f_attn, f_hopf, ctx_bits
    )
    return jnp.sum(E_b)


# ------------------------------------------------------------
# Inference: forward Euler on activation-gradient flow
# ------------------------------------------------------------
def _init_hidden(params, v_0, ctx_bits):
    """
    Initialize hidden states at their targets for v_0.
    """
    xi_attn_emb = get_xi_attn_emb(params)  # (vocab_size, D)
    xi_attn_seq = xi_attn_emb[ctx_bits]  # (B, L, D)
    h_att_0 = jnp.einsum("bld,bd->bl", xi_attn_seq, v_0) + params["b"]  # (B, L)

    xi_hopf = get_xi_hopf(params)  # (M, D)
    H_hopf0 = v_0 @ xi_hopf.T + params["c"]  # (B, M)
    return h_att_0, H_hopf0

# TODO: unify this with the other infer_forward_euler function
@functools.partial(jax.jit, donate_argnums=(1,))  # donate v_0 buffer
def infer_forward_euler_with_force(params, v_0, ctx_bits):
    """
    Returns:
      v_T: (B, D) terminal visible state
      F_T: (B, D) force at v_T, i.e. dV/dt = -(1/tau_v) * dE/dV at v_T
    """
    xi_attn_emb = get_xi_attn_emb(params)  # (vocab_size, D)
    xi_hopf = get_xi_hopf(params)

    xi_attn_seq = xi_attn_emb[ctx_bits]  # (B, L, D)

    step_v = Config.step_size / Config.tau_v
    step_h = Config.step_size / Config.tau_h

    # Initial pre-activations
    h_att_0 = jnp.einsum("bld,bd->bl", xi_attn_seq, v_0) + params["b"]
    h_hopf_0 = v_0 @ xi_hopf.T + params["c"]

    # Energy for batch
    def energy_batch(params, V, h_attn, h_hopf, f_attn, f_hopf):
        def energy_per_sample(v, h_att, h_hopf, f_attn, f_hopf, xi_attn):
            dv = v - params["a"]
            vis = 0.5 * jnp.dot(dv, dv)
            coupling = jnp.dot(v, xi_attn.T @ f_attn + xi_hopf.T @ f_hopf)
            att_bias = jnp.dot(f_attn, h_att - params["b"])
            hopf_bias = jnp.dot(f_hopf, h_hopf - params["c"])
            return (
                vis - coupling + att_bias + hopf_bias - L_attn(h_att) - L_hopf(h_hopf)
            )

        Eb = jax.vmap(energy_per_sample, in_axes=(0, 0, 0, 0, 0, 0))(
            V, h_attn, h_hopf, f_attn, f_hopf, xi_attn_seq
        )
        return jnp.sum(Eb)

    grad_E = jax.grad(energy_batch, argnums=(1, 4, 5))  # grads wrt (V, f_attn, f_hopf)

    def grads_activation(V, h_attn, h_hopf):
        f_attn = jax.nn.softmax(Config.beta * h_attn, axis=-1)
        f_hopf = jnp.maximum(h_hopf, 0.0)
        dE_dV, dE_df_attn, dE_df_hopf = grad_E(params, V, h_attn, h_hopf, f_attn, f_hopf)
        return dE_dV, dE_df_attn, dE_df_hopf

    def body(_, carry):
        V, h_attn, h_hopf = carry
        dE_dV, dE_df_attn, dE_df_hopf = grads_activation(V, h_attn, h_hopf)
        V = V - step_v * dE_dV
        h_attn = h_attn - step_h * dE_df_attn
        h_hopf = h_hopf - step_h * dE_df_hopf
        return V, h_attn, h_hopf

    v_T, H_att_T, H_hopf_T = jax.lax.fori_loop(
        0, Config.n_steps, body, (v_0, h_att_0, h_hopf_0)
    )

    # Force on visible neurons at final state (no extra allocations beyond one grad eval)
    dE_dV_T, _, _ = grads_activation(v_T, H_att_T, H_hopf_T)
    F_T = -(1.0 / Config.tau_v) * dE_dV_T
    return v_T, F_T


@jax.jit
def infer_forward_euler(params, v_0, ctx_bits):
    """
    Returns:
      v_T: (B, D)
      traj: dict of optional trajectories (currently V only)
    """
    xi_attn_emb = get_xi_attn_emb(params)  # (vocab_size, D)
    xi_hopf = get_xi_hopf(params)

    xi_attn_seq = xi_attn_emb[ctx_bits]  # (B, L, D)
    h_att_0 = jnp.einsum("bld,bd->bl", xi_attn_seq, v_0) + params["b"]
    h_hopf_0 = v_0 @ get_xi_hopf(params).T + params["c"]

    # Pack everything needed for grads to avoid repeated closures
    def energy_batch_w_Xi(params, V, h_attn, h_hopf, f_attn, f_hopf):
        # same as energy_per_batch, but use xi_attn_seq captured from outer scope
        def energy_per_sample_wXi(v, h_attn, h_hopf, f_attn, f_hopf, xi_attn):
            dv = v - params["a"]
            vis = 0.5 * jnp.dot(dv, dv)
            coupling = jnp.dot(v, xi_attn.T @ f_attn + xi_hopf.T @ f_hopf)
            att_bias = jnp.dot(f_attn, h_attn - params["b"])
            hopf_bias = jnp.dot(f_hopf, h_hopf - params["c"])
            return vis - coupling + att_bias + hopf_bias - L_attn(h_attn) - L_hopf(h_hopf)

        Eb = jax.vmap(energy_per_sample_wXi, in_axes=(0, 0, 0, 0, 0, 0))(
            V, h_attn, h_hopf, f_attn, f_hopf, xi_attn_seq
        )
        return jnp.sum(Eb)

    grad_E = jax.grad(energy_batch_w_Xi, argnums=(1, 4, 5))  # grads w.r.t. (V, f_attn, f_hopf)

    def grads_activation(V, h_attn, h_hopf):
        f_attn = jax.nn.softmax(Config.beta * h_attn, axis=-1)
        f_hopf = jnp.maximum(h_hopf, 0.0)
        (
            dE_dV,
            dE_df_attn,
            dE_df_hopf,
        ) = grad_E(params, V, h_attn, h_hopf, f_attn, f_hopf)
        return dE_dV, dE_df_attn, dE_df_hopf, f_attn, f_hopf

    def step(carry, _):
        V, h_attn, h_hopf = carry
        dE_dV, dE_df_attn, dE_df_hopf, f_attn, f_hopf = grads_activation(V, h_attn, h_hopf)
        V_n = V - (Config.step_size / Config.tau_v) * dE_dV
        H_att_n = h_attn - (Config.step_size / Config.tau_h) * dE_df_attn
        H_hopf_n = h_hopf - (Config.step_size / Config.tau_h) * dE_df_hopf
        return (V_n, H_att_n, H_hopf_n), V_n

    (v_T, _, _), V_traj = jax.lax.scan(
        step, (v_0, h_att_0, h_hopf_0), xs=None, length=Config.n_steps
    )
    return v_T, V_traj


# ------------------------------------------------------------
# Direct inference with full trajectories
# ------------------------------------------------------------
def run_model_inference_steps(ctx_bits, params, v_0=None):
    """
    Run forward Euler inference and return trajectories of
    logits, visible V, hidden preactivations (h_attn, h_hopf),
    and per-sample mixed energy for all steps.

    Shapes:
      ctx_bits: (B, L)
      v_0 (optional): (B, D), defaults to zeros
    """
    B = ctx_bits.shape[0]
    if v_0 is None:
        v_0 = jnp.zeros((B, Config.D), dtype=jnp.float32)

    h_att_0, h_hopf_0 = _init_hidden(params, v_0, ctx_bits)

    def step(carry, _):
        V, h_attn, h_hopf = carry

        # Activations at current state (used only for gradients)
        f_attn = jax.nn.softmax(Config.beta * h_attn, axis=-1)  # (B, L)
        f_hopf = jnp.maximum(h_hopf, 0.0)  # (B, M)

        # Gradients wrt activation-coordinates
        dE_dV = jax.grad(energy_per_batch, argnums=1)(
            params, V, h_attn, h_hopf, f_attn, f_hopf, ctx_bits
        )
        dE_df_attn = jax.grad(energy_per_batch, argnums=4)(
            params, V, h_attn, h_hopf, f_attn, f_hopf, ctx_bits
        )
        dE_df_hopf = jax.grad(energy_per_batch, argnums=5)(
            params, V, h_attn, h_hopf, f_attn, f_hopf, ctx_bits
        )

        # Forward Euler updates
        V_n = V - (Config.step_size / Config.tau_v) * dE_dV
        h_att_n = h_attn - (Config.step_size / Config.tau_h) * dE_df_attn
        h_hopf_n = h_hopf - (Config.step_size / Config.tau_h) * dE_df_hopf

        # Log post-update quantities (time n=t+Δt)
        f_attn_n = jax.nn.softmax(Config.beta * h_att_n, axis=-1)  # (B, L)
        f_hopf_n = jnp.maximum(h_hopf_n, 0.0)  # (B, M)
        logits_n = logits_from_v(params, V_n)  # (B, 2)

        # Per-sample mixed energy at post-update state
        E_b = jax.vmap(energy_per_sample, in_axes=(None, 0, 0, 0, 0, 0, 0))(
            params, V_n, h_att_n, h_hopf_n, f_attn_n, f_hopf_n, ctx_bits
        )  # (B,)

        return (V_n, h_att_n, h_hopf_n), (V_n, h_att_n, h_hopf_n, logits_n, E_b)

    # Scan for T steps; collect trajectories
    (v_T, h_att_T, h_hopf_T), (v_traj, h_att_traj, h_hopf_traj, logits_traj, E_traj) = (
        jax.lax.scan(step, (v_0, h_att_0, h_hopf_0), xs=None, length=Config.n_steps)
    )

    traj = dict(
        v=v_traj,  # (T, B, D)
        h_attn=h_att_traj,  # (T, B, L)
        h_hopf=h_hopf_traj,  # (T, B, M)
        logits=logits_traj,  # (T, B, 2)
        energy=E_traj,  # (T, B)  per-sample energies
    )
    return (v_T, h_att_T, h_hopf_T), traj


# ------------------------------------------------------------
# Readout and loss
# ------------------------------------------------------------
def logits_from_v(params: ModelParams, V: jax.Array) -> jax.Array:
    return V @ params["w_dec"].T + params["b_dec"]  # (B, 2)


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
    v_0 = jnp.zeros((B, Config.D))  # visible init
    v_T, F_T = infer_forward_euler_with_force(params, v_0, ctx_bits)
    logits_T = logits_from_v(params, v_T)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits_T, labels).mean()
    return ce + force_weight * force_penalty(F_T)


@jax.jit
def evaluate(params: ModelParams, valid_X: jax.Array, valid_y: jax.Array) -> jax.Array:
    v_T, _ = infer_forward_euler_with_force(
        params, jnp.zeros((valid_y.shape[0], Config.D), jnp.float32), valid_X
    )
    preds = jnp.argmax(logits_from_v(params, v_T), axis=1)
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
            name = last.key  # 'xi_attn_emb', 'xi_hopf', etc.
        else:
            name = str(last)
            
        if name in ("xi_attn_emb", "xi_hopf"):
            print('fast param:', name)
            return "fast"
        else:
            print('slow param:', name)
            return "slow"

    return jax.tree_util.tree_map_with_path(label_from_path, params)
