from typing import Literal
import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from mamba3.nn.ssm import ssm_forward
from mamba3.nn.linear import MLP
from mamba3.nn.norm import RMSNorm

@dataclasses.dataclass(frozen=True)
class Mamba3_Config:
    hidden_dims: int = 512
    state_dims: int = 64
    heads: int = 8
    rank_size: int = 4
    chunk_size: int = 8

    mlp_inner_dims: int = 896
    mlp_act_fn: Literal["silu", "gelu", "relu", "tanh"] = "silu"

    num_mamba_layers: int = 6

    use_bias: bool = True

class Mamba3(nnx.Module):
    def __init__(
        self,
        cfg: Mamba3_Config,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_dims = cfg.hidden_dims
        self.state_dims = cfg.state_dims
        self.heads = cfg.heads
        self.rank_size = cfg.rank_size
        self.chunk_size = cfg.chunk_size

        self.norm = RMSNorm(self.rank_size, rngs=rngs)
        
        self.in_proj = nnx.Linear(
            self.hidden_dims,
            (self.heads * self.rank_size) + self.state_dims + 
             2 * (self.state_dims * self.rank_size) + self.state_dims // 2 + 2,
            use_bias=False,
            rngs=rngs
        )

        self.resid_proj = nnx.Linear(
            self.hidden_dims,
            self.heads * self.rank_size,
            use_bias=True,
            rngs=rngs
        )

        self.out_proj = nnx.Linear(
            self.heads * self.rank_size,
            self.hidden_dims,
            use_bias=False,
            rngs=rngs
        )

        self.b_bias = nnx.Param(
            jnp.ones((self.state_dims, self.rank_size))
        )

        self.c_bias = nnx.Param(
            jnp.ones((self.state_dims, self.rank_size))
        )

    def generate_init_state_and_input(self, batch_size):
        state = jnp.zeros((batch_size, self.state_dims, self.heads)) # (B, D, H)
        x = jnp.zeros((batch_size, self.heads, self.rank_size)) # (B, H, R)
        b = jnp.zeros((batch_size, self.state_dims, self.rank_size)) # (B, D, R)
        dt = jnp.broadcast_to(jnp.array([1e-4]), shape=(batch_size, self.state_dims,)) #(D,)
        return state, x, b, dt

    def __call__(self, x: jax.Array):
        B, L, _ = x.shape

        if L < self.chunk_size:
            chunk_size = L
        else:
            chunk_size = self.chunk_size

        init_state, init_x, init_b, _ = self.generate_init_state_and_input(B)

        state, y = ssm_forward(self.in_proj, x, self.b_bias.value, 
            self.c_bias.value, self.norm, init_state, init_x, init_b,
            self.heads, self.rank_size, chunk_size)

        resid_x = self.resid_proj(x)
        resid_x = resid_x.reshape(B, L, self.heads, self.rank_size)

        y = jnp.multiply(y, nnx.sigmoid(resid_x))
        y = y.reshape(B, L, self.rank_size * self.heads)
        y = self.out_proj(y)

        return state, y
        
class Mamba3Model(nnx.Module):
    """
        Pure mamba backbone model.
    """
    def __init__(
            self,
            mamba_cfg: Mamba3_Config, 
            num_layers: int, 
            *, 
            rngs: nnx.Rngs
        ):
        super().__init__()
        self.layers = nnx.List(
            [Mamba3(mamba_cfg, rngs=rngs) for _ in range(num_layers)]
        )

        self.padding_token = nnx.Param(
            nnx.initializers.normal(stddev=1.0)(rngs(), (mamba_cfg.hidden_dims))
        )

    def __call__(self, x: jax.Array):
        for layer in self.layers:
            _, x = layer(x)
        return x
    
class Mamba3MLP(nnx.Module):
    """
        Mamba 3 model with MLP interleaved.
    """
    def __init__(
            self,
            cfg: Mamba3_Config,
            *,
            rngs: nnx.Rngs):
        super().__init__()

        self.hidden_dims = cfg.hidden_dims
        self.num_layers = cfg.num_mamba_layers * 2 + 1

        layers = []
        for i in range(self.num_layers):
            layers.append(
                Mamba3(cfg, rngs=rngs) if i % 2 == 0 else 
                MLP(cfg.hidden_dims, cfg.mlp_inner_dims, cfg.mlp_act_fn, rngs=rngs)
            )
        self.layers = nnx.List(layers)

        self.padding_token = nnx.Param(
            nnx.initializers.normal(stddev=1.0)(rngs(), (cfg.hidden_dims,))
        )

        self.norm = RMSNorm(self.hidden_dims, rngs=rngs)
        
    def __call__(self, x: jax.Array):
        y = jnp.zeros_like(x)
        for layer in self.layers:
            if type(layer) == Mamba3:
                y = y + layer(self.norm(x))[1]
            else:
                y = y + layer(self.norm(x))
        return x
    