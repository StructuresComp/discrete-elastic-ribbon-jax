"""Banded linear solver via jax.pure_callback -> scipy.linalg.solve_banded.

LAPACK banded storage convention:
    ab[k + i - j, j] = A[i, j]   for max(0, j-k) <= i <= min(N-1, j+k)
    ab.shape == (2k+1, N)

The solve returns NaN on singular matrices; callers handle fallback (escalated
regularisation) in the outer Newton retry loop.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as _sla


def _scipy_banded_solve_host(ab, b, k):
    try:
        return _sla.solve_banded((int(k), int(k)), ab, b).astype(b.dtype)
    except (np.linalg.LinAlgError, ValueError):
        return np.full_like(b, np.nan)


def banded_solve(ab: jax.Array, b: jax.Array, k: int) -> jax.Array:
    """Solve A x = b where A is banded (l=u=k), stored in LAPACK format ab.

    Dispatches to scipy.linalg.solve_banded on the host via jax.pure_callback.
    Returns NaN vector on singular systems.
    """
    N = b.shape[0]
    out_shape = jax.ShapeDtypeStruct((N,), b.dtype)
    return jax.pure_callback(
        partial(_scipy_banded_solve_host, k=k),
        out_shape,
        ab, b,
    )


def banded_to_dense(ab: jax.Array, k: int) -> jax.Array:
    """Unpack LAPACK-banded storage back to a dense (N, N) matrix.

    Used only for diagnostic / SVD-fallback paths.
    """
    N = ab.shape[1]
    i_idx = jnp.arange(N)[:, None]
    j_idx = jnp.arange(N)[None, :]
    band_row = k + i_idx - j_idx
    in_band = (band_row >= 0) & (band_row <= 2 * k)
    safe_band_row = jnp.where(in_band, band_row, 0)
    values = ab[safe_band_row, j_idx]
    return jnp.where(in_band, values, 0.0)
