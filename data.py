import os
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def generate_and_save_datasets(
    key: jax.Array,
    ctx_length: int = 8,
    train_ratio: float = 0.8,
    filename_prefix: str = "parity_data",
):
    n = 2**ctx_length
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

    train_X = bits[: int(n * train_ratio)]
    train_y = labels[: int(n * train_ratio)]
    test_X = bits[int(n * train_ratio) :]
    test_y = labels[int(n * train_ratio) :]

    # Save to disk as txt
    os.makedirs("data", exist_ok=True)
    np.savetxt(f"data/{filename_prefix}_train_X.txt", train_X, fmt="%d")
    np.savetxt(f"data/{filename_prefix}_train_y.txt", train_y, fmt="%d")
    np.savetxt(f"data/{filename_prefix}_test_X.txt", test_X, fmt="%d")
    np.savetxt(f"data/{filename_prefix}_test_y.txt", test_y, fmt="%d")


def load_dataset(
    filename_prefix: str = "parity_data",
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    train_X = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_train_X.txt", dtype=np.int32)
    )
    train_y = jnp.array(
        np.loadtxt(f"data/{filename_prefix}_train_y.txt", dtype=np.int32)
    )
    test_X = jnp.array(np.loadtxt(f"data/{filename_prefix}_test_X.txt", dtype=np.int32))
    test_y = jnp.array(np.loadtxt(f"data/{filename_prefix}_test_y.txt", dtype=np.int32))
    return train_X, train_y, test_X, test_y
