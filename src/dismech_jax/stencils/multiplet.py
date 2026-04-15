"""Multiplet stencil — 5-node window composed of 3 adjacent Triplets.

A `Multiplet` centred at node i covers nodes {i-2, i-1, i, i+1, i+2}, total
19 DOFs:

    [n_{i-2}, θ_{i-2},  n_{i-1}, θ_{i-1},  n_i, θ_i,  n_{i+1}, θ_{i+1},  n_{i+2}]

The three overlapping 11-DOF Triplet windows are:

    sub-triplet k-1   → indices  0..10   (centred at i-1)
    sub-triplet k     → indices  4..14   (centred at i)
    sub-triplet k+1   → indices  8..18   (centred at i+1)

`get_strain` returns a 12-component vector concatenating the 5-component
triplet strains of those three sub-triplets, minus their stored bar_strains:

    multiplet strain = [η̂_{i-1}, η̂_i, η̂_{i+1}]                 (size 3*4=12 after
                                                                  dropping the
                                                                  duplicate ε)

Actually we keep the full 5-component strain per sub-triplet (5+5+5 = 15) —
higher-level energy models decide how to reduce / reuse. Downstream code can
slice to the canonical 4-component η̂ per triplet (ε̄, κ₁, κ₂, τ) as needed.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from .stencil import Stencil
from .triplet import Triplet
from ..states import TripletState


class Multiplet(Stencil):
    """3-triplet stencil over 5 consecutive nodes (19 local DOFs, 15-dim strain)."""

    # Three adjacent triplets as a batched pytree (leading dim 3).
    sub_triplets: Triplet                 # batched over 3

    @classmethod
    def init_from_triplets(cls, triplets_batch: Triplet, aux_batch: TripletState,
                            q_multi: jax.Array) -> "Multiplet":
        """Construct a Multiplet from a batched Triplet pytree (already sliced
        along axis 0 to 3 adjacent triplets) and the local 19-DOF q_multi."""
        mp = cls(
            bar_strain=jnp.empty(0),
            sub_triplets=triplets_batch,
        )
        bar = mp.get_strain(q_multi, aux_batch)
        return cls(bar_strain=bar, sub_triplets=triplets_batch)

    @staticmethod
    def slice_to_sub_q(q_multi: jax.Array) -> jax.Array:
        """Split 19-DOF q_multi into 3 overlapping 11-DOF sub-triplet windows."""
        starts = jnp.array([0, 4, 8])
        return jax.vmap(
            lambda s: jax.lax.dynamic_slice(q_multi, (s,), (11,))
        )(starts)                                     # (3, 11)

    def get_strain(self, q: jax.Array, aux: TripletState | None = None) -> jax.Array:
        """Per-Multiplet strain: concatenate 3 sub-triplet strains → (15,)."""
        sub_qs = self.slice_to_sub_q(q)               # (3, 11)
        strains = jax.vmap(
            lambda st, qq, a: st.get_strain(qq, a)
        )(self.sub_triplets, sub_qs, aux)             # (3, 5)
        return strains.ravel()                        # (15,)
