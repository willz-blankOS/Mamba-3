from typing import Literal, Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from mamba3.nn.norm import RMSNorm

def compute_discrete_rotary_blocks(disc_freq: jax.Array):
    assert disc_freq.shape[-1] % 2 == 0, "Last dimension of x must be even for rotary embeddings."
    
    disc_freq = disc_freq[..., None] # Shape: (B, L, H/2, 1)
    cos_emb = jnp.cos(disc_freq)
    sin_emb = jnp.sin(disc_freq)
    disc_freq_upper = jnp.concatenate([cos_emb, -sin_emb], axis=-1) # Shape: (B, L, H/2, 2)
    disc_freq_lower = jnp.concatenate([sin_emb, cos_emb], axis=-1) # Shape: (B, L, H/2, 2)
    disc_freq = jnp.stack([disc_freq_upper, disc_freq_lower], axis=-1) # Shape: (B, L, H/2, 2, 2)

    return disc_freq

def apply_rotation(x: jax.Array, rotatary_blocks: jax.Array):
    shape = x.shape
    assert x.shape[-2] % 2 == 0, f"Dimension must be a factor of 2, recieved {x.shape[-1]}"

    x = x.transpose(0, 2, 1)
    
    x1, x2 = jnp.split(x, 2, axis=-1)
    x_blocks = jnp.stack([x1, x2], axis=-1)[...,None,:] # (B, R, D//2, 1, 2)

    x = x_blocks @ rotatary_blocks[:,None,...]

    # Apply rotatory block
    x = x.transpose(0, 2, 1, 3, 4)
    x = x.reshape(*shape)    
    return x

def ssm_chunked_forward(
    x: jax.Array,
    last_x: jax.Array,
    z: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    last_B: jax.Array,
    theta: jax.Array,
    dt: jax.Array,
    lam: jax.Array,
    last_cum_rotation: jax.Array
):
    """
    Compute the forward pass of a state space model (SSM) in a chunked manner.
    This function computes the forward pass of a state space model (SSM) using the provided
    parameters A, B, C, theta, dt, and lam. The computation is performed in chunks of size block_size
    
    Args:
        x (jax.Array): The input sequence. (B, L, H, R)
        last_x (jax.Array): The input sequence. (B, H, R)
        z (jax.Array): The initial state of the SSM. (B, D, H)
        A (jax.Array): The state transition matrix. (B, L, D)
        B (jax.Array): The expansion matrix. (B, L, D, H)
        theta (jax.Array): The data dependent theta for complex transition and RoPE. (B, L, D // 2)
        dt (jax.Array): The time step for the SSM. (B, L, 1)
        lam (jax.Array): The lambda parameter for the SSM. (B, L, 1)
        block_size (int): The size of the chunks to process at a time. Default is
        first (bool): First chuck in the sequence
    """
    batch_size, _, D = A.shape
    alpha = jnp.exp(A * dt)
    beta = (1 - lam) * dt * jnp.exp(A * dt)
    gamma = lam * dt
    rotary_disc_blocks = compute_discrete_rotary_blocks(dt * theta)
    rotary_disc_blocks_t = jnp.transpose(rotary_disc_blocks, (0, 1, 2, 4, 3)) # Shape: (B, L, D // 2, 2, 2)

    def forward(carry, inputs):
        """
            last_x: (B, H, R)
            last_z: (B, D, H)
            x: (B, H, R)
            b: (B, D, H)
            alpha: (B, D, 1)
            beta: (B, D, 1)
            gamma: (B, D, 1)
            cum_rotation: (B, D//2, 2, 2)
        """
        last_z, _last_cum_rotation, last_b = carry
        x, b, c, alpha, beta, gamma, rotary_disc_blocks_t = inputs

        curr_cum_rotation = jnp.einsum(
            "...ij,...jk->...ik", _last_cum_rotation, rotary_disc_blocks_t
        )
        
        # Compute the next state z_next using the SSM equations
        z_a = alpha[...,None] * last_z
        b_m1 = beta[...,None] * jnp.einsum(
            "...ij,...jk->...ik",
            apply_rotation(last_b, last_cum_rotation), 
            jnp.transpose(last_x, (0, 2, 1)
        ))
        b_t = gamma[...,None] * jnp.einsum(
            "...ij,...jk->...ik",
            apply_rotation(b, curr_cum_rotation),
            jnp.transpose(x, (0, 2, 1)
        ))
        curr_z = z_a + b_m1 + b_t
        c_rotated = apply_rotation(c, curr_cum_rotation)
        c_rotated_t = c_rotated.transpose(0, 2, 1)
        y = jnp.einsum("...ij,...jk->...ik", c_rotated_t, curr_z)
        return (curr_z, curr_cum_rotation, b), y

    carry = (z, last_cum_rotation, last_B)
    inputs=(x, B, C, alpha, beta, gamma, rotary_disc_blocks_t)
    
    (latest_state, latest_rotation, latest_b), y = nnx.scan(
        forward, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1)
    )(carry, inputs)
    
    return (latest_state, latest_rotation, latest_b), y
    
def sequencial_chunk(x: jax.Array, block_size: int, axis: int):
    axis_size = x.shape[axis]
    chunk_size = axis_size // block_size
    
    x_chunks = jnp.array_split(x, chunk_size, axis=axis)
    return x_chunks

def ssm_forward(
    in_proj: nnx.Linear, 
    x: jax.Array, # (B, L, C)
    b_bias: jax.Array, # (D,R)
    c_bias: jax.Array, # (D,R)
    norm: RMSNorm,
    init_state: jax.Array, # (D, H)
    init_x: jax.Array, # (H, R)
    init_b: jax.Array, # (D, R)
    head_size: int,
    rank_size: int,
    block_size: int,
):
    B, L, _ = x.shape
    state_size = init_state.shape[-2]
    xABCtdl = in_proj(x)
    
    xABCtdl_projected = jnp.split(
        xABCtdl, [
        head_size * rank_size, 
        (head_size * rank_size) + state_size, 
        (head_size * rank_size) + state_size + (state_size * rank_size),
        (head_size * rank_size) + state_size + 2 * (state_size * rank_size), 
        (head_size * rank_size) + state_size + 2 * (state_size * rank_size) + state_size // 2, 
        (head_size * rank_size) + state_size + 2 * (state_size * rank_size) + state_size // 2 + 1], 
        axis=-1
    )
    x, a, b, c, theta, dt, lam = xABCtdl_projected
    
    x = x.reshape(B, L, head_size, rank_size) # Shape: (B, L, H, R)
    a = a.reshape(B, L, state_size) # Shape: (B, L, D)
    b = b.reshape(B, L, state_size, rank_size) # (B, L, D, R)
    c = c.reshape(B, L, state_size, rank_size) # (B, L, D, R)
    theta = theta.reshape(B, L, state_size // 2) # (B, L, D//2)
    dt = jax.nn.sigmoid(dt.reshape(B, L, 1)) * 0.1 + 1e-4 # (B, L, 1)
    lam = jax.nn.sigmoid(lam.reshape(B, L, 1)) # (B, L, 1)

    # Normalize expansion and contraction
    b = norm(b) + b_bias[None,:,:]
    c = norm(c) + c_bias[None,:,:]
    
    x_chunked = sequencial_chunk(x, block_size, axis=1)
    a_chunked = sequencial_chunk(a, block_size, axis=1)
    b_chunked = sequencial_chunk(b, block_size, axis=1)
    c_chunked = sequencial_chunk(c, block_size, axis=1)
    theta_chunked = sequencial_chunk(theta, block_size, axis=1)
    dt_chunked = sequencial_chunk(dt, block_size, axis=1)
    lam_chunked = sequencial_chunk(lam, block_size, axis=1)
    
    # Chunk inputs
    def forward(carry, chnks):
        z, last_x, last_rotation, last_b = carry
        x_chunk, a_chunk, b_chunk, c_chunk, theta_chunk, dt_chunk, lam_chunk = chnks
        
        (curr_z, cum_rotation, curr_b), y = ssm_chunked_forward(
            x_chunk, last_x, z, a_chunk, b_chunk, c_chunk, last_b, 
            theta_chunk, dt_chunk, lam_chunk, last_cum_rotation=last_rotation,
        )
        
        return (curr_z, x_chunk[:,-1,:], cum_rotation, curr_b), y

    init_rotation = (jnp.identity(2))[None, None,:]
    init_rotation = init_rotation.repeat(state_size//2, 1).repeat(B, axis=0)
    last_carry = (
        init_state, 
        init_x,
        init_rotation,
        init_b
    )
    y_list = []
    for i, chnk in enumerate(zip(x_chunked, a_chunked, b_chunked, c_chunked, 
                                 theta_chunked, dt_chunked, lam_chunked)):
        carry, y = forward(last_carry, chnk)
        last_carry = carry
        y_list.append(y)

    y = jnp.concatenate(y_list, axis=1).reshape(B, L, head_size, rank_size)
    return last_carry[0], y
    