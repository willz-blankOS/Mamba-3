"""
Mamba3 JAX/Flax NNX Implementation.

A pure JAX/Flax implementation of the Mamba3 architecture using the State Space Duality (SSD) mechanism.
Reference: "Mamba-3: Improved Sequence Modeling using State Space Principles"
Paper: https://arxiv.org/abs/2603.15569
"""


from typing import Literal
import dataclasses

import jax
import flax.nnx as nnx
