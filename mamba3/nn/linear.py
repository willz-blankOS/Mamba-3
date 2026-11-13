from typing import Literal
import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx

@dataclasses.dataclass(frozen=True)
class MLP_Config:
    hidden_dims: int = 512
    inner_dims: int = 786
    act_fn: Literal["silu", "gelu", "relu", "tanh"] = "silu"

ACT2FN = {
    "silu": nnx.silu,
    "gelu": nnx.gelu,
    "relu": nnx.relu,
    "tanh": nnx.tanh
}

class MLP(nnx.Module):
    def __init__(
            self, 
            hidden_dims: int, 
            inner_dims: int, 
            act_fn: Literal["silu", "gelu", "relu", "tanh"] = "silu",
            *, 
            rngs: nnx.Rngs
        ):
        super().__init__()

        self.in_proj = nnx.Linear(
            hidden_dims,
            inner_dims,
            use_bias=True,
            rngs=rngs
        )

        self.act_fn = ACT2FN[act_fn]

        self.out_proj = nnx.Linear(
            inner_dims,
            hidden_dims,
            use_bias=True,
            rngs=rngs
        )

    def __call__(self, x: jax.Array):
        x = self.in_proj(x)
        x = self.act_fn(x)
        x = self.out_proj(x)
        return x
