import os
from typing import List

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax

# from config import Config
from data import generate_and_save_datasets, load_dataset
from model import evaluate, force_penalty_weight, init_params, label_tree, loss_fn
from utils import save_metrics, save_params


def stack_grad_trees(grad_trees):
    """
    grad_trees: list of PyTrees with identical structure
    returns: PyTree where each leaf is stacked over axis 0
    """
    return jtu.tree_map(
        lambda *xs: np.stack(xs, axis=0),
        *grad_trees,
    )


if __name__ == "__main__":
    with open("output.txt", "w") as f:
        for attn_scale in [0.1]:
            for hopf_scale in [0.06]:
                print(
                    f"attn_scale: {attn_scale}, hopf_scale: {hopf_scale}",
                    file=f,
                    flush=True,
                )

                from config import Config

                Config.xi_attn_embed_raw_scale = attn_scale
                Config.xi_hopf_raw_scale = hopf_scale
                print(
                    f"{Config.xi_attn_embed_raw_scale}, {Config.xi_hopf_raw_scale}",
                    file=f,
                    flush=True,
                )

                key = jr.PRNGKey(Config.seed)

                # TODO: use different key
                generate_and_save_datasets(
                    key=key,
                    ctx_length=Config.L,
                    train_ratio=0.8,
                    filename_prefix="parity_data",
                )
                train_X, train_y, valid_X, valid_y = load_dataset(
                    filename_prefix="parity_data"
                )

                params = init_params(key)

                num_train = train_X.shape[0]
                num_batches = (num_train + Config.batch_size - 1) // Config.batch_size
                total_steps = Config.train_epochs * num_batches

                def lr_sched(
                    peak: float, warmup_steps: int = 0, end_factor: float = 0.3
                ) -> optax.Schedule:
                    """warm up to peak, then decay to peak * end_factor"""
                    return optax.warmup_cosine_decay_schedule(
                        init_value=Config.lr_init_value,
                        peak_value=peak,
                        warmup_steps=warmup_steps,
                        decay_steps=max(1, total_steps - warmup_steps),
                        end_value=peak * end_factor,
                    )

                # don't apply weight decay to xi_attn_embed_raw and xi_hopf_raw
                tx_fast = optax.chain(
                    optax.clip_by_global_norm(Config.max_norm),
                    optax.adamw(
                        learning_rate=lr_sched(Config.lr_peak_value),
                        weight_decay=Config.fast_weight_decay,
                    ),
                )
                tx_slow = optax.chain(
                    optax.clip_by_global_norm(Config.max_norm),
                    optax.adamw(
                        learning_rate=lr_sched(Config.lr_peak_value),
                        weight_decay=Config.slow_weight_decay,
                    ),
                )

                optimizer = optax.multi_transform(
                    {"fast": tx_fast, "slow": tx_slow}, label_tree(params)
                )
                opt_state = optimizer.init(params)

                @jax.jit
                def train_step(params, opt_state, train_X, train_Y, force_weight):
                    loss, grads = jax.value_and_grad(loss_fn)(
                        params, train_X, train_Y, force_weight
                    )
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    return params, opt_state, loss

                losses_all = []
                losses_steps: List[int] = []
                accs_all = []
                accs_steps: List[int] = []
                # grads_all = []
                for epoch in range(Config.train_epochs):
                    t_start = time.time()
                    key, key_perm = jr.split(key)
                    index_perm = jr.permutation(key_perm, num_train)
                    train_X_epoch = train_X[index_perm]
                    train_y_epoch = train_y[index_perm]
                    losses_epoch = []

                    # compute current force penalty weight (cosine ramp 0 -> 1)
                    lam_force = jnp.asarray(
                        force_penalty_weight(epoch), dtype=jnp.float32
                    )

                    for batch in range(num_batches):
                        start = batch * Config.batch_size
                        stop = min((batch + 1) * Config.batch_size, num_train)
                        key, sub = jr.split(key)
                        batch_train_X = train_X_epoch[start:stop]
                        batch_train_y = train_y_epoch[start:stop]

                        params, opt_state, loss = train_step(
                            params,
                            opt_state,
                            batch_train_X,
                            batch_train_y,
                            force_weight=lam_force,
                        )
                        losses_epoch.append(loss)
                        losses_all.append(float(loss))
                        losses_steps.append(epoch * num_batches + batch)
                        # if epoch < 2000:
                        # grads_all.append(grads)
                        if jnp.isnan(loss):
                            break

                    if jnp.isnan(loss):
                        break

                    t_end = time.time()

                    if epoch % 100 == 0:
                        acc = evaluate(params, valid_X, valid_y)
                        accs_all.append(float(acc))
                        accs_steps.append((epoch + 1) * num_batches - 1)
                        print(
                            f"epoch {epoch:5d} | "
                            f"force_w {float(lam_force):5.3f} | "
                            f"train loss {jnp.mean(jnp.array(losses_epoch)):8.4f} | "
                            f"valid acc {acc:6.4f} | "
                            f"{(t_end - t_start):3.3f}s / epoch | ",
                            file=f,
                            flush=True,
                        )
                        save_params(params, "data/model.npz")

                        if acc >= 0.99:
                            save_params(params, "data/model_best.npz")

                save_metrics(
                    {"step": losses_steps, "loss": losses_all}, "data/losses.json"
                )
                save_metrics(
                    {"step": accs_steps, "accuracy": accs_all}, "data/accs.json"
                )

                save_params(params, "data/model.npz")
                print(
                    "Training complete. Model parameters saved to 'data/model.npz'.",
                    file=f,
                    flush=True,
                )

                # grads_stacked = stack_grad_trees(
                #     [jtu.tree_map(lambda x: np.asarray(x), g) for g in grads_all]
                # )

                # np.savez(
                #     "data/grads_batch0_all_epochs.npz",
                #     **grads_stacked,
                # )
