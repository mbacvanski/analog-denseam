"""
Train the Analog ET model using energy-based training on the parity dataset.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools
import pickle
import time
import math  # for cosine schedule

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
class Cfg:
    D = 16
    L = 8
    M = 16
    beta = 0.1

    # --- Stable integration parameters ---
    step_size = 1e-3
    T_final = 1.0
    n_steps = int(T_final / step_size)

    batch_size = 256
    train_epochs = 30_000
    seed = 0
    vocab_size = 2

    tau_v = 1e-1
    tau_h = 1e-2


# ------------------------------------------------------------
# Params (NO W_q, W_k; we train xi directly via an embedding)
# ------------------------------------------------------------
def init_params(key):
    k1, k2, k3 = jr.split(key, 3)

    # Token -> conductance row mapping (per-sequence weights)
    # xi_emb[t] \in R^D gives the row xi_j used for token t
    xi_emb = jr.normal(k1, (Cfg.vocab_size, Cfg.D)) * 0.1

    # Per-position attention bias b_j (length L)
    b_att = jnp.zeros((Cfg.L,))

    # Hopfield memory (rows eta_m) and thresholds c_m
    # eta = jr.normal(k2, (Cfg.M, Cfg.D)) * (1.0 / jnp.sqrt(Cfg.D))
    eta = jr.normal(k2, (Cfg.M, Cfg.D)) * 0.06
    c = jnp.zeros((Cfg.M,))

    # Visible bias a_v
    a_v = jnp.zeros((Cfg.D,))

    # Readout for parity classification
    W_out = jr.normal(k3, (2, Cfg.D)) * (1.0 / jnp.sqrt(Cfg.D))
    b_out = jnp.zeros((2,))

    return dict(xi_emb=xi_emb, b_att=b_att, eta=eta, c=c, a_v=a_v, W_out=W_out, b_out=b_out)


# ------------------------------------------------------------
# Positive-use wrappers (store raw, use clamped)
# ------------------------------------------------------------
def xi_pos(params):
    # Positive version of xi_emb used in all computations
    # return jnp.maximum(params["xi_emb"], 0.0)
    return jnp.square(params["xi_emb"])


def eta_pos(params):
    # Positive version of eta used in all computations
    # return jnp.maximum(params["eta"], 0.0)
    return jnp.square(params["eta"])


# ------------------------------------------------------------
# Data: Parity (context of L bits; label = XOR of bits)
# ------------------------------------------------------------
def sample_batch(key, X, Y, batch_size):
    indices = jr.choice(key, X.shape[0], shape=(batch_size,), replace=False)
    ctx_bits = X[indices].astype(jnp.int32)
    labels = Y[indices].astype(jnp.int32)
    return ctx_bits, labels


def L_attn(h):  # (B,L) -> (B,)
    return (1.0 / Cfg.beta) * jax.nn.logsumexp(Cfg.beta * h, axis=-1)


def L_hopf(h):  # (B,M) -> (B,)
    r = jnp.maximum(h, 0.0)
    return 0.5 * jnp.sum(r * r, axis=-1)


# ------------------------------------------------------------
# Mixed-coordinate energy
# ------------------------------------------------------------
def energy_per_sample(params, v, h_att, h_hopf, p, a, ctx_bits_row):
    # Xi: (L, D) rows selected by the tokens in this sequence (positive-used)
    Xi_all = xi_pos(params)  # (vocab_size, D)
    Xi = Xi_all[ctx_bits_row]  # (L, D)

    # Visible quadratic
    dv = v - params['a_v']
    vis_term = 0.5 * jnp.dot(dv, dv)  # scalar

    # Couplings (positive-used eta)
    XiTp = Xi.T @ p  # (D,)
    etaTa = eta_pos(params).T @ a  # (D,)
    coupling = jnp.dot(v, XiTp + etaTa)  # scalar

    # Linear Fenchel-Young-style saddle terms (activation coords)
    att_bias = jnp.dot(p, h_att - params['b_att'])  # scalar
    hopf_bias = jnp.dot(a, h_hopf - params['c'])  # scalar

    # Total mixed-coordinate energy
    return vis_term - coupling + att_bias + hopf_bias - L_attn(h_att) - L_hopf(h_hopf)


def energy_per_batch(params, V, H_att, H_hopf, P, A, ctx_bits):
    """
    V:      (B, D)
    H_att:  (B, L)
    H_hopf: (B, M)
    P:      (B, L)   -- activations (softmax of H_att), treated as independent arg
    A:      (B, M)   -- activations (ReLU of H_hopf), treated as independent arg
    ctx_bits: (B, L)
    """
    E_b = jax.vmap(energy_per_sample, in_axes=(None, 0, 0, 0, 0, 0, 0))(params, V, H_att, H_hopf, P, A, ctx_bits)
    return jnp.sum(E_b)


# ------------------------------------------------------------
# Inference: forward Euler on activation-gradient flow
# ------------------------------------------------------------
def _init_hidden(params, V0, ctx_bits):
    """
    Initialize hidden states at their targets for V0.
    H_att0 = Xi V0 + b_att
    H_hopf0 = eta V0 + c
    """
    # Xi_seq: (B, L, D) using positive-used xi
    Xi_all = xi_pos(params)  # (vocab_size, D)
    Xi_seq = Xi_all[ctx_bits]  # (B, L, D)
    H_att0 = jnp.einsum('bld,bd->bl', Xi_seq, V0) + params['b_att']  # (B, L)

    # Positive-used eta
    H_hopf0 = V0 @ eta_pos(params).T + params['c']  # (B, M)
    return H_att0, H_hopf0


@functools.partial(jax.jit, donate_argnums=(1,))  # donate V0 buffer
def infer_forward_euler_with_force(params, V0, ctx_bits):
    """
    Returns:
      V_T: (B, D) terminal visible state
      F_T: (B, D) force at V_T, i.e. dV/dt = -(1/tau_v) * dE/dV at V_T
    """
    # Xi_seq with positive-used xi
    Xi_all = xi_pos(params)  # (vocab_size, D)
    Xi_seq = Xi_all[ctx_bits]  # (B, L, D)
    step_v = Cfg.step_size / Cfg.tau_v
    step_h = Cfg.step_size / Cfg.tau_h

    # Initial pre-activations
    H_att0 = jnp.einsum('bld,bd->bl', Xi_seq, V0) + params['b_att']
    H_hopf0 = V0 @ eta_pos(params).T + params['c']

    # Energy for batch with Xi captured (no per-step allocations beyond carry)
    def energy_batch_w_Xi(params, V, H_att, H_hopf, P, A):
        def energy_per_sample_wXi(v, h_att, h_hopf, p, a, Xi_row):
            dv = v - params['a_v']
            vis = 0.5 * jnp.dot(dv, dv)
            XiTp = Xi_row.T @ p  # (D,)
            etaTa = eta_pos(params).T @ a  # (D,)
            coupling = jnp.dot(v, XiTp + etaTa)
            att_bias = jnp.dot(p, h_att - params['b_att'])
            hopf_bias = jnp.dot(a, h_hopf - params['c'])
            return vis - coupling + att_bias + hopf_bias - L_attn(h_att) - L_hopf(h_hopf)

        Eb = jax.vmap(
            energy_per_sample_wXi, in_axes=(0, 0, 0, 0, 0, 0)
        )(V, H_att, H_hopf, P, A, Xi_seq)
        return jnp.sum(Eb)

    grad_E = jax.grad(energy_batch_w_Xi, argnums=(1, 4, 5))  # grads wrt (V, P, A)

    def grads_activation(V, H_att, H_hopf):
        P = jax.nn.softmax(Cfg.beta * H_att, axis=-1)
        A = jnp.maximum(H_hopf, 0.0)
        dE_dV, dE_dP, dE_dA = grad_E(params, V, H_att, H_hopf, P, A)
        return dE_dV, dE_dP, dE_dA

    def body(_, carry):
        V, H_att, H_hopf = carry
        dE_dV, dE_dP, dE_dA = grads_activation(V, H_att, H_hopf)
        V = V - step_v * dE_dV
        H_att = H_att - step_h * dE_dP
        H_hopf = H_hopf - step_h * dE_dA
        return (V, H_att, H_hopf)

    V_T, H_att_T, H_hopf_T = jax.lax.fori_loop(
        0, Cfg.n_steps, body, (V0, H_att0, H_hopf0)
    )

    # Force at terminal state (no extra allocations beyond one grad eval)
    dE_dV_T, _, _ = grads_activation(V_T, H_att_T, H_hopf_T)
    F_T = -(1.0 / Cfg.tau_v) * dE_dV_T
    return V_T, F_T


@jax.jit
def infer_forward_euler(params, V0, ctx_bits):
    """
    Returns:
      V_T: (B, D)
      traj: dict of optional trajectories (currently V only)
    """
    B = V0.shape[0]
    Xi_all = xi_pos(params)  # (vocab_size, D)
    Xi_seq = Xi_all[ctx_bits]  # (B, L, D)
    H_att0 = jnp.einsum('bld,bd->bl', Xi_seq, V0) + params['b_att']
    H_hopf0 = V0 @ eta_pos(params).T + params['c']

    # Pack everything needed for grads to avoid repeated closures
    def energy_batch_w_Xi(params, V, H_att, H_hopf, P, A):
        # same as energy_per_batch, but use Xi_seq captured from outer scope
        def energy_per_sample_wXi(v, h_att, h_hopf, p, a, Xi_row):
            dv = v - params['a_v']
            vis = 0.5 * jnp.dot(dv, dv)
            XiTp = Xi_row.T @ p  # (D,)
            etaTa = eta_pos(params).T @ a  # (D,)
            coupling = jnp.dot(v, XiTp + etaTa)
            att_bias = jnp.dot(p, h_att - params['b_att'])
            hopf_bias = jnp.dot(a, h_hopf - params['c'])
            return vis - coupling + att_bias + hopf_bias - L_attn(h_att) - L_hopf(h_hopf)

        Eb = jax.vmap(energy_per_sample_wXi, in_axes=(0, 0, 0, 0, 0, 0))(V, H_att, H_hopf, P, A, Xi_seq)
        return jnp.sum(Eb)

    grad_E = jax.grad(energy_batch_w_Xi, argnums=(1, 4, 5))  # grads w.r.t. (V, P, A)

    def grads_activation(V, H_att, H_hopf):
        P = jax.nn.softmax(Cfg.beta * H_att, axis=-1)
        A = jnp.maximum(H_hopf, 0.0)
        dE_dV, dE_dP, dE_dA, = grad_E(params, V, H_att, H_hopf, P, A)
        return dE_dV, dE_dP, dE_dA, P, A

    def step(carry, _):
        V, H_att, H_hopf = carry
        dE_dV, dE_dP, dE_dA, P, A = grads_activation(V, H_att, H_hopf)
        Vn = V - (Cfg.step_size / Cfg.tau_v) * dE_dV
        H_att_n = H_att - (Cfg.step_size / Cfg.tau_h) * dE_dP
        H_hopf_n = H_hopf - (Cfg.step_size / Cfg.tau_h) * dE_dA
        return (Vn, H_att_n, H_hopf_n), Vn

    (V_T, _, _), V_traj = jax.lax.scan(step, (V0, H_att0, H_hopf0), xs=None, length=Cfg.n_steps)
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
    if V0 is None:
        V0 = jnp.zeros((B, Cfg.D), dtype=jnp.float32)

    H_att0, H_hopf0 = _init_hidden(params, V0, ctx_bits)

    def step(carry, _):
        V, H_att, H_hopf = carry

        # Activations at current state (used only for gradients)
        P = jax.nn.softmax(Cfg.beta * H_att, axis=-1)  # (B, L)
        A = jnp.maximum(H_hopf, 0.0)  # (B, M)

        # Gradients wrt activation-coordinates
        dE_dV = jax.grad(energy_per_batch, argnums=1)(params, V, H_att, H_hopf, P, A, ctx_bits)
        dE_dP = jax.grad(energy_per_batch, argnums=4)(params, V, H_att, H_hopf, P, A, ctx_bits)
        dE_dA = jax.grad(energy_per_batch, argnums=5)(params, V, H_att, H_hopf, P, A, ctx_bits)

        # Forward Euler updates
        Vn = V - (Cfg.step_size / Cfg.tau_v) * dE_dV
        H_att_n = H_att - (Cfg.step_size / Cfg.tau_h) * dE_dP
        H_hopf_n = H_hopf - (Cfg.step_size / Cfg.tau_h) * dE_dA

        # Log post-update quantities (time t+Δt)
        Pn = jax.nn.softmax(Cfg.beta * H_att_n, axis=-1)  # (B, L)
        An = jnp.maximum(H_hopf_n, 0.0)  # (B, M)
        logits_n = logits_from_v(params, Vn)  # (B, 2)

        # Per-sample mixed energy at post-update state
        E_b = jax.vmap(
            energy_per_sample,
            in_axes=(None, 0, 0, 0, 0, 0, 0)
        )(params, Vn, H_att_n, H_hopf_n, Pn, An, ctx_bits)  # (B,)

        return (Vn, H_att_n, H_hopf_n), (Vn, H_att_n, H_hopf_n, logits_n, E_b)

    # Scan for T steps; collect trajectories
    (V_T, H_att_T, H_hopf_T), (V_traj, H_att_traj, H_hopf_traj, logits_traj, E_traj) = jax.lax.scan(
        step, (V0, H_att0, H_hopf0), xs=None, length=Cfg.n_steps
    )

    traj = dict(
        V=V_traj,  # (T, B, D)
        H_att=H_att_traj,  # (T, B, L)
        H_hopf=H_hopf_traj,  # (T, B, M)
        logits=logits_traj,  # (T, B, 2)
        energy=E_traj,  # (T, B)  per-sample energies
    )
    return (V_T, H_att_T, H_hopf_T), traj


# ------------------------------------------------------------
# Readout and loss
# ------------------------------------------------------------
def logits_from_v(params, V):
    return V @ params['W_out'].T + params['b_out']  # (B, 2)


def _force_penalty(F_T):
    return jnp.mean(jnp.sum(F_T * F_T, axis=1))  # MSE of force


@jax.jit
def loss_fn(params, key, ctx_bits, labels, force_weight):
    """
    force_weight: scalar multiplier for the force penalty term.
    """
    B = ctx_bits.shape[0]
    V0 = jnp.zeros((B, Cfg.D))  # visible init
    V_T, F_T = infer_forward_euler_with_force(params, V0, ctx_bits)
    logits_T = logits_from_v(params, V_T)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits_T, labels).mean()
    return ce + force_weight * _force_penalty(F_T)


# ------------------------------------------------------------
# Training step
# ------------------------------------------------------------
@jax.jit
def train_step(params, opt_state, train_X, train_Y, key, force_weight):
    k1, k2 = jr.split(key)
    loss, grads = jax.value_and_grad(loss_fn)(params, k2, train_X, train_Y, force_weight)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


def generate_and_save_datasets(key, ctx_length=8, train_ratio=0.8, filename_prefix="parity_data"):
    n = 2 ** ctx_length
    # Create numbers 0 .. n - 1
    ints = jnp.arange(n, dtype=jnp.uint32)
    # Extract bits by shifting and masking
    bits = (ints[:, None] >> jnp.arange(ctx_length - 1, -1, -1)) & 1
    bits = bits.astype(jnp.int16)

    # Labels are parity (XOR of bits)
    labels = jnp.sum(bits, axis=1) % 2
    labels = labels.astype(jnp.int16)

    # shuffle bits and labels
    perm_key, _ = jr.split(key)
    perm = jr.permutation(perm_key, n)
    bits = bits[perm]
    labels = labels[perm]

    train_X = bits[:int(n * train_ratio)]
    train_y = labels[:int(n * train_ratio)]
    test_X = bits[int(n * train_ratio):]
    test_y = labels[int(n * train_ratio):]

    # Save to disk as txt
    np.savetxt(f"{filename_prefix}_train_X.txt", train_X, fmt='%d')
    np.savetxt(f"{filename_prefix}_train_y.txt", train_y, fmt='%d')
    np.savetxt(f"{filename_prefix}_test_X.txt", test_X, fmt='%d')
    np.savetxt(f"{filename_prefix}_test_y.txt", test_y, fmt='%d')


def load_dataset(filename_prefix="parity_data"):
    train_X = jnp.array(np.loadtxt(f"{filename_prefix}_train_X.txt", dtype=np.int32))
    train_y = jnp.array(np.loadtxt(f"{filename_prefix}_train_y.txt", dtype=np.int32))
    test_X = jnp.array(np.loadtxt(f"{filename_prefix}_test_X.txt", dtype=np.int32))
    test_y = jnp.array(np.loadtxt(f"{filename_prefix}_test_y.txt", dtype=np.int32))
    return train_X, train_y, test_X, test_y


@jax.jit
def evaluate(params, valid_X, valid_y):
    V_T, _ = infer_forward_euler_with_force(params, jnp.zeros((valid_y.shape[0], Cfg.D), jnp.float32), valid_X)
    preds = jnp.argmax(logits_from_v(params, V_T), axis=1)
    return jnp.mean((preds == valid_y).astype(jnp.float32))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    key = jr.PRNGKey(Cfg.seed)

    generate_and_save_datasets(key=key, ctx_length=Cfg.L, train_ratio=0.8, filename_prefix="parity_data")
    train_X, train_y, valid_X, valid_y = load_dataset(filename_prefix="parity_data")

    params = init_params(key)

    total_steps = Cfg.train_epochs * ((train_X.shape[0] + Cfg.batch_size - 1) // Cfg.batch_size)  # ~epochs * steps/epoch


    # Decay learning rate to 0.3 by the end of training.
    def lr_sched(peak, warm=0, end=0.3):
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=peak,
            warmup_steps=warm,
            decay_steps=max(1, total_steps - warm),
            end_value=peak * end
        )


    # Around 17k epochs the model starts to get 100% accuracy on the validation set, at which point start ramping up the force penalty.
    # Applying force penalty too early causes the model to never reach good accuracies.
    def force_penalty_weight(epoch: int) -> float:
        start = 20_000
        duration = 10_000
        end = start + duration  # = 30_000

        if epoch < start:
            return 0.0
        if epoch >= end:
            return 1.0

        # normalized progress in [0,1]
        t = (epoch - start) / duration

        # cosine ramp: 0 → 1
        return 0.05 * (1.0 - math.cos(math.pi * t))


    # don't apply weight decay to xi_emb and eta
    tx_fast = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(learning_rate=lr_sched(3e-3), weight_decay=0))
    tx_slow = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.adamw(learning_rate=lr_sched(3e-3), weight_decay=5e-5))

    from jax.tree_util import DictKey


    # Parameter partition
    def label_tree(params):
        def label_from_path(path, leaf):
            last = path[-1]
            if isinstance(last, DictKey):
                name = last.key  # 'xi_emb', 'eta', etc.
            else:
                name = str(last)
            if name in ("xi_emb", "eta"):
                return "fast"
            else:
                return "slow"

        return jax.tree_util.tree_map_with_path(label_from_path, params)


    optimizer = optax.multi_transform({"fast": tx_fast, "slow": tx_slow}, label_tree(params))
    opt_state = optimizer.init(params)

    num_train = train_X.shape[0]
    num_batches = (num_train + Cfg.batch_size - 1) // Cfg.batch_size

    losses_all = []
    losses_steps = []
    accs_all = []
    accs_steps = []
    for epoch in range(Cfg.train_epochs):
        t_start = time.time()
        key, key_perm = jr.split(key)
        index_perm = jr.permutation(key_perm, num_train)
        train_X_epoch = train_X[index_perm]
        train_y_epoch = train_y[index_perm]
        losses_epoch = []

        # compute current force penalty weight (cosine ramp 0 -> 1)
        lam_force = jnp.asarray(force_penalty_weight(epoch), dtype=jnp.float32)

        for batch in range(num_batches):
            start = batch * Cfg.batch_size
            stop = min((batch + 1) * Cfg.batch_size, num_train)
            key, sub = jr.split(key)
            batch_train_X = train_X_epoch[start:stop]
            batch_train_y = train_y_epoch[start:stop]

            params, opt_state, loss = train_step(
                params, opt_state, batch_train_X, batch_train_y, key=sub, force_weight=lam_force
            )
            losses_epoch.append(loss)
            losses_all.append(float(loss))
            losses_steps.append(epoch * num_batches + batch)

        t_end = time.time()

        if epoch % 100 == 0:
            acc = evaluate(params, valid_X, valid_y)
            # accs.append(acc)
            accs_all.append(float(acc))
            accs_steps.append((epoch + 1) * num_batches - 1)
            print(f"epoch {epoch:5d} | "
                  f"force_w {float(lam_force):5.3f} | "
                  f"train loss {jnp.mean(jnp.array(losses_epoch)):8.4f} | "
                  f"valid acc {acc:6.4f} | "
                  f"{(t_end - t_start):3.3f}s / epoch | "
                  )
            with open("model.pkl", "wb") as f:
                pickle.dump(params, f)

            if acc >= 0.99:
                with open("model_best.pkl", "wb") as f:
                    pickle.dump(params, f)

    with open("losses.pkl", "wb") as f:
        pickle.dump((losses_steps, losses_all), f)
    with open("accs.pkl", "wb") as f:
        pickle.dump((accs_steps, accs_all), f)

    with open("model.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Training complete. Model parameters saved to 'model.pkl'.")
