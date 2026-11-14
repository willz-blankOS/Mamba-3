from typing import Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from mamba3.nn.norm import RMSNorm

def compute_discrete_rotary_blocks(disc_freq: jax.Array):
    assert disc_freq.shape[-1] % 2 == 0, "Last dimension of x must be even for rotary embeddings."
    
    disc_freq = disc_freq[..., None] # Shape: (B, L, H/2, 1)
    cos_emb = jnp.cos(disc_freq)
    sin_emb = jnp.sin(disc_freq)
    disc_freq_upper = jnp.concatenate([cos_emb, -sin_emb], axis=-1)[..., None] # Shape: (B, L, H/2, 2)
    disc_freq_lower = jnp.concatenate([sin_emb, cos_emb], axis=-1)[..., None] # Shape: (B, L, H/2, 2)
    disc_freq = jnp.stack([disc_freq_upper, disc_freq_lower], axis=-1) # Shape: (B, L, H/2, 2, 2)

    return disc_freq

def apply_rotation(x: jax.Array, rotatary_blocks: jax.Array):
    B, L, H, D = x.shape
    assert x.shape[-1] % 2 == 0, f"Dimension must be a factor of 2, recieved {D}"
    
    x1, x2 = jnp.split(x, 2, axis=-1)
    x_blocks = jnp.stack([x1, x2], axis=-1)
    
    # Apply rotatory block
    x_blocks = jnp.einsum("...ij,...jk->...ik", x_blocks, rotatary_blocks)
    x = x.reshape(B, L, H, D)
    
    return x

def ssm_chunked_forward(
    x: jax.Array,
    last_x: jax.Array,
    z: jax.Array,
    A: jax.Array,
    B: jax.Array,
    last_B: jax.Array,
    C: jax.Array,
    last_cum_rotation: jax.Array,
    theta: jax.Array,
    dt: jax.Array,
    lam: jax.Array
):
    """
    Compute the forward pass of a state space model (SSM) in a chunked manner.
    This function computes the forward pass of a state space model (SSM) using the provided
    parameters A, B, C, theta, dt, and lam. The computation is performed in chunks of size block_size
    
    Args:
        x (jax.Array): The input sequence. (B, L, D)
        last_x (jax.Array): The input sequence. (B, 1, D)
        z (jax.Array): The initial state of the SSM. (B, D)
        A (jax.Array): The state transition matrix. (B, L, D)
        B (jax.Array): The expansion matrix. (B, L, D)
        B (jax.Array): The expansion matrix. (B, 1, D)
        C (jax.Array): The contraction matrix. (B, L, D)
        theta (jax.Array): The data dependent theta for complex transition and RoPE. (B, L, D // 2)
        dt (jax.Array): The time step for the SSM. (B, L)
        lam (jax.Array): The lambda parameter for the SSM. (B, L)
        block_size (int): The size of the chunks to process at a time. Default is
    """
    
    alpha = jnp.exp(A * dt)
    beta = (1 - lam) * dt * jnp.exp(A * dt)
    gamma = lam * dt
    
    rotary_disc_blocks = compute_discrete_rotary_blocks(dt * theta)
    rotary_disc_blocks_t = jnp.transpose(rotary_disc_blocks, (0, 1, 2, 4, 3)) # Shape: (B, L, D // 2, 2, 2)
    
    def matmul_op(a, b):
        return jnp.einsum("bhij,bhjk->bhik", a, b)
    
    cum_rotations = jax.lax.associative_scan(
        matmul_op, 
        jnp.concatenate([last_cum_rotation, rotary_disc_blocks_t], axis=1), 
        axis=1
    )[:, 1:]
    
    def forward(carry, inputs):
        last_z = carry
        x, last_x, b, last_b, c, alpha, beta, gamma, last_cum_rotation, cum_rotation = inputs
        
        # Compute the next state z_next using the SSM equations
        z_a = alpha * last_z
        b_m1 = beta * (apply_rotation(last_b, last_cum_rotation) @ last_x)
        b_t = gamma * (apply_rotation(b, cum_rotation) @ x)
        curr_z = z_a + b_m1 + b_t
        
        return (curr_z, cum_rotation, b), curr_z
    
    last_x_init = jnp.concatenate([last_x, x[..., :-1]], axis=1)
    x_init = x
    last_b_init = jnp.concatenate([last_B, B[..., :-1]], axis=1)
    b_init = B
    last_cum_rotation_init = jnp.concatenate([last_cum_rotation, cum_rotations[..., :-1]], axis=1)
    cum_rotation_init = cum_rotations
    (latest_state, latest_rotation, latest_b), states = jax.lax.scan(
        forward, z, (x_init, last_x_init, b_init, last_b_init, C, 
         alpha, beta, gamma, last_cum_rotation_init, cum_rotation_init)
    )
    
    return (latest_state, latest_rotation, latest_b), states
    
def sequencial_chunk(x: jax.Array, block_size: int, axis: int):
    axis_size = x.shape[axis]
    assert axis_size % block_size == 0, "Block size not compatible with sequence" 
    num_chunks = axis_size // block_size
    
    x_chunks = jnp.array_split(x, num_chunks, axis=axis)
    return x_chunks

def ssm_forward(
    in_proj: nnx.Linear, 
    x: jax.Array, # (B, L, C)
    b_bias: jax.Array, # (H,D)
    c_bias: jax.Array, # (P,D)
    norm: RMSNorm,
    init_state: jax.Array, # (H, D)
    init_x: jax.Array, # (H, D)
    init_b: jax.Array, # (H, D)
    pre_head_size: int,
    post_head_size: int,
    block_size: int,
):
    B, L, _ = x.shape
    state_size = init_state.shape[-1]
    xABC = in_proj(x)
    
    x, a, b, c, theta, dt, lam = jnp.split(
        xABC, 
        [pre_head_size * state_size,
         state_size,
         pre_head_size * state_size,
         post_head_size * state_size,
         state_size // 2,
         pre_head_size,
         pre_head_size], 
        axis=-1
    )
    
    x = x.reshape(B, L, pre_head_size, state_size)
    a = a.reshape(B, L, state_size)
    b = b.reshape(B, L, pre_head_size, state_size)
    c = c.reshape(B, L, post_head_size, state_size)
    theta = theta.reshape(B, L, state_size // 2)
    dt = dt.reshape(B, L, 1)
    lam = lam.reshape(B, L, 1)
    
    b = norm(b + b_bias[None,None,:])
    c = norm(c + c_bias[None,None,:])
    
    x_chunked = sequencial_chunk(x, block_size, axis=1)
    a_chunked = sequencial_chunk(a, block_size, axis=1)
    b_chunked = sequencial_chunk(b, block_size, axis=1)
    c_chunked = sequencial_chunk(c, block_size, axis=1)
    theta_chunked = sequencial_chunk(theta, block_size, axis=1)
    dt_chunked = sequencial_chunk(dt, block_size, axis=1)
    lam_chunked = sequencial_chunk(lam, block_size, axis=1)
    
    # Slice initial values
    init_x = x[:,:,0:0,:]
    init_b = b[:,:,0:0,:]
    init_theta = theta[:,:,0:0,:]
    init_dt = dt[:,0:0]
    
    # Chunk inputs
    def forward(carry, chnks):
        z, last_x, last_b, last_rotation = carry
        x_chunk, a_chunk, b_chunk, c_chunk, theta_chunk, dt_chunk, lam_chunk = chnks
        
        (curr_z, curr_b, cum_rotation), states = ssm_chunked_forward(
            x_chunk, last_x, z, a_chunk, b_chunk, last_b, c_chunk, 
            last_rotation, theta_chunk, dt_chunk, lam_chunk, 
            block_size=block_size
        )

        curr_c_rotated = apply_rotation(c_chunk, cum_rotation)
        curr_c_rotated_t = jnp.transpose(curr_c_rotated[...,None], axes=(0, 1, 2, 4, 3))
        y = jnp.einsum(
            "...ij,...jk->...ik", curr_c_rotated_t,
            jnp.sum(states, axis=2)[...,None]
        )
        
        return (curr_z, curr_b, cum_rotation), y
    
    # Iterate over chunks 
    #(state, _, _, _), y_chnk = jax.lax.scan(
    #    forward, init_carry, 
    #    (x_chunked, a_chunked, b_chunked, c_chunked,
    #     theta_chunked, dt_chunked, lam_chunked)
    #)


    init_rotation = compute_discrete_rotary_blocks(init_dt * init_theta)
    last_carry = (init_state, init_x, init_b, init_rotation)
    y_list = []
    for _, chnk in enumerate(zip(x_chunked, a_chunked, b_chunked, c_chunked, 
        theta_chunked, dt_chunked, lam_chunked)
    ):
        carry, y = forward(last_carry, chnk)
        last_carry = carry
        y_list.append(y)

    y = jnp.stack(y_list, axis=1).reshape(B, L, post_head_size, state_size)
    
    return last_carry[0], y
    