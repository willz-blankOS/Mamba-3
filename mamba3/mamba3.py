from typing import Literal
import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from nn.ssm import ssm_forward
from nn.linear import MLP, MLP_Config

@dataclasses.dataclass(frozen=True)
class Mamba3_Config:
    hidden_dims: int = 512
    state_size: int = 128
    pre_heads: int = 4
    post_heads: int = 4
    chunk_size: int = 4

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
        self.pre_heads = cfg.pre_heads
        self.post_heads = cfg.post_heads
        self.chunk_size = cfg.chunk_size
        
        self.in_proj = nnx.Linear(
            self.hidden_dims,
            (2 * (self.pre_heads * self.state_dims) + (self.post_heads * self.state_dims)
             + self.state_dims + self.state_dims // 2 + 2 * self.pre_heads),
            use_bias=False,
            rngs=rngs
        )

        self.resid_proj = nnx.Linear(
            self.hidden_dims,
            self.post_heads * self.state_dims,
            rngs=rngs
        )

        self.out_proj = nnx.Linear(
            self.post_heads * self.state_dims,
            self.hidden_dims,
            rngs=rngs
        )

        self.b_bias = nnx.Param(
            jnp.ones((self.pre_heads, self.state_dims))
        )

        self.c_bias = nnx.Param(
            jnp.ones((self.post_heads, self.state_dims))
        )

    def generate_init_state_and_input(self):
        state = jnp.zeros((self.pre_heads, self.state_dims))
        x = jnp.zeros((self.pre_heads, self.state_dims))
        b = jnp.zeros((self.pre_heads, self.state_dims))
        return state, x, b

    def __call__(self, x: jax.Array):
        B, L, D = x.shape

        init_state, init_x, init_b = self.generate_init_state_and_input()

        state, y = ssm_forward(self.in_proj, x, self.b_bias.value, 
            self.c_bias.value, self.norm, init_x, init_state, init_b,
            self.pre_heads, self.post_heads, self.chunk_size)

        resid_x = self.resid_proj(x[:,-1,:])
        resid_x = resid_x.reshape(B, self.post_heads, self.state_dims)

        y = jnp.multiply(y, nnx.silu(resid_x))
        y = y.reshape(B, L, self.post_heads * self.state_dims)
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
        
    def __call__(self, x: jax.Array):
        for layer in self.layers:
            _, x = layer(x)
        return x
