
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
import numpy as np
import jax.random as random
from jax import vmap

import numpyro
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
from typing import Callable


# --- General Thermodynamic Integration (TI) ---
# We consider a positive family g(Z, r) with log_g(Z, r) = log g(Z, r).
# Define c(r) = ∫ p0(Z) g(Z, r) dZ.
# Then d/dr log c(r) = E_r[ ∂/∂r log g(Z, r) ].
# We estimate this expectation via MCMC at grid points r and integrate numerically.

TI_SITE = "_TI_Z"
"""Name of the deterministic site holding the sampled `Z`, read back after sampling."""

def model_for_ti(
    r: float,
    log_g_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    prior: Callable[[], jnp.ndarray],
) -> None:
    """NumPyro model of the deformed prior at a fixed `r`.

    The deformed density is `p_r(Z) = p0(Z) * g(Z, r) / c(r)`, and this model defines
    it up to the normalization constant `c(r)`: `Z` is drawn from the base prior `p0`
    and `log g(Z, r)` is added as a factor.

    Args:
        r: the deformation parameter ("inverse temperature")
        log_g_fn: maps `(Z, r)` to the unnormalized log factor `log g(Z, r)`; must be
            JAX-compatible and differentiable with respect to `r`
        prior: samples `Z` from the base prior `p0`
    """
    Z = prior()
    numpyro.factor("log_g", log_g_fn(Z, r))

    # Recorded as a deterministic so the samples can be retrieved from the trace
    numpyro.deterministic(TI_SITE, Z)

def run_thermodynamic_integration(
    rng_key: jax.Array,
    log_g_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    prior: Callable[[], jnp.ndarray],
    r_grid: jax.Array,
    num_samples: int,
    num_warmup: int | None = None,
    progress: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Estimate the log normalization constant `log c(r)` by thermodynamic integration.

    For each `r` in `r_grid`, samples are drawn from the deformed prior
    `p_r(Z) ∝ p0(Z) g(Z, r)` and used to estimate `E_r[∂/∂r log g(Z, r)]`, which is the
    derivative of `log c(r)`. Integrating these estimates over `r_grid` gives `log c(r)`
    up to an additive constant.

    Each grid point is warm-started from the sampler state of the previous one, so the
    chains do not have to re-adapt from scratch.

    Args:
        rng_key: JAX random key
        log_g_fn: maps `(Z, r)` to `log g(Z, r)`; must be differentiable w.r.t. `r`
        prior: samples `Z` from the base prior `p0`
        r_grid: increasing grid of `r` values
        num_samples: post-warmup samples per grid point
        num_warmup: warmup steps for NUTS; defaults to `10 + 0.2 * num_samples`
        progress: whether to show a progress bar over the grid points

    Returns:
        The estimated derivatives `E_r[∂/∂r log g(Z, r)]` at each grid point, and the
        resulting `log c(r)`, anchored at `log c(r_grid[0]) = 0`.
    """
    if num_warmup is None:
        num_warmup = 10 + int(0.2 * num_samples)

    # One key for the initial adaptation run, one per grid point
    keys = random.split(rng_key, len(r_grid) + 1)
    warmup_key, grid_keys = keys[0], keys[1:]

    def dlogg_dr(Z: jnp.ndarray, r: float) -> jnp.ndarray:
        """Derivative of `log g(Z, r)` with respect to `r`, for a single sample."""
        return jax.grad(lambda r_: log_g_fn(Z, r_))(r)

    dlogg_dr_batched = vmap(dlogg_dr, in_axes=(0, None))

    kernel = NUTS(model_for_ti, find_heuristic_step_size=True)
    mcmc_args = dict(num_warmup=num_warmup, num_chains=4, progress_bar=False)

    # Adapt the sampler once at the first grid point, with a longer run
    num_init_samples = int(num_samples * len(r_grid) / 2)
    mcmc = MCMC(kernel, num_samples=num_init_samples, **mcmc_args)
    mcmc.run(warmup_key, r=r_grid[0], log_g_fn=log_g_fn, prior=prior)
    mcmc_state = mcmc.last_state

    derivative_expectations = []
    grid = tqdm(r_grid, desc="Thermodynamic integration", disable=not progress)
    for i, r in enumerate(grid):
        mcmc = MCMC(kernel, num_samples=num_samples, **mcmc_args)
        mcmc.post_warmup_state = mcmc_state
        mcmc.run(grid_keys[i], r=r, log_g_fn=log_g_fn, prior=prior)

        Z_samples = mcmc.get_samples()[TI_SITE]
        derivative_expectations.append(jnp.mean(dlogg_dr_batched(Z_samples, r)))
        mcmc_state = mcmc.last_state

    dlogc_dr_values = jnp.stack(derivative_expectations)

    # log c(r) = ∫ E_r[∂_r log g(Z, r)] dr, anchored at the first grid point
    log_c_values = jnp.array(
        cumulative_trapezoid(np.asarray(dlogc_dr_values), np.asarray(r_grid), initial=0)
    )

    return dlogc_dr_values, log_c_values

def estimate_marginal_log_likelihood_curve(
    base_prior: Callable[[], jnp.ndarray],
    log_g_fn: Callable[[jnp.ndarray, float], jnp.ndarray],
    r_grid: jax.Array,
    rng_key: jax.Array | None = None,
    num_samples: int = 5000,
    num_warmup: int | None = None,
    progress: bool = False,
) -> tuple[RegularGridInterpolator, jax.Array, jax.Array]:
    """Estimate `log c(r)` over a grid and return a differentiable interpolator.

    Runs thermodynamic integration over `r_grid` and wraps the result in an
    interpolator, so that `log c(r)` can be evaluated at a sampled `r` inside a model.
    The values are shifted so that `log c(r)` is 0 at the largest grid point; the
    constant is arbitrary, since only differences in `log c` affect the posterior.

    Args:
        base_prior: samples `Z` from the base prior `p0`
        log_g_fn: maps `(Z, r)` to `log g(Z, r)`
        r_grid: the grid of `r` values to estimate over
        rng_key: JAX random key; drawn at random if not given
        num_samples: post-warmup samples per grid point
        num_warmup: warmup steps for NUTS
        progress: whether to show a progress bar over the grid points

    Returns:
        An interpolator over `log c(r)`, the estimated derivatives, and the `log c(r)`
        values on the (ascending) grid.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(np.random.randint(2 ** 31, dtype=np.uint32))

    dlogc_dr_values, log_c_values = run_thermodynamic_integration(
        rng_key=rng_key,
        log_g_fn=log_g_fn,
        prior=base_prior,
        r_grid=r_grid,
        num_samples=num_samples,
        num_warmup=num_warmup,
        progress=progress,
    )

    # The interpolator requires an ascending grid
    order = jnp.argsort(r_grid)
    r_grid = r_grid[order]
    log_c_values = log_c_values[order]

    # Fix the arbitrary constant by anchoring the curve at the largest r
    log_c_values -= log_c_values[-1]

    interpolator = RegularGridInterpolator((r_grid,), log_c_values)

    return interpolator, dlogc_dr_values, log_c_values
