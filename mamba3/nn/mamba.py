import jax
import jax.numpy as jnp
import flax.nnx as nnx

class Mamba3(nnx.Module):
    def __init__(
        self,  
        *, 
        rngs: nnx.Rngs
    ):
        ...
        
    @property
    def get_config():
        return {
            
        }
        
class Mamba3Block(nnx.Module):
    def __init__(
        self, 
        *, 
        rngs: nnx.Rngs
    ):
        ...
        
class Mamba3(nnx.Module):
    ...
    
class Mamba3MLP(nnx.Module):
    ...