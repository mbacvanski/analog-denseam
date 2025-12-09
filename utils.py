import json
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

ModelParams = Dict[str, jax.Array]


def save_params(params: ModelParams, fname: str):
    np.savez(fname, **{k: np.array(v) for k, v in params.items()})  # type: ignore


def load_params(fname: str) -> ModelParams:
    loaded = np.load(fname)
    return {k: jnp.array(loaded[k]) for k in loaded.files}


def save_metrics(obj: Dict[str, List[int | float]], fname: str):
    with open(fname, "w") as f:
        json.dump(obj, f)


def load_metrics(fname: str) -> dict:
    with open(fname, "r") as f:
        data = json.load(f)
    return data
