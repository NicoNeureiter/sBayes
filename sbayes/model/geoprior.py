from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from numpyro.distributions import Distribution
from scipy.integrate import cumulative_trapezoid
import numpy as np
import jax.random as random
from jax import vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# For plotting the results
import matplotlib.pyplot as plt


# --- General Thermodynamic Integration (TI) ---
# We consider a positive family g(Z, r) with log_g(Z, r) = log g(Z, r).
# Define c(r) = ∫ p0(Z) g(Z, r) dZ.
# Then d/dr log c(r) = E_r[ ∂/∂r log g(Z, r) ].
# We estimate this expectation via MCMC at grid points r and integrate numerically.


def model_for_ti(
        r: float,
        log_g_fn: callable,
        prior: callable,
):
    """NumPyro model for a fixed r with unnormalized log factor log_g(Z, r).

    The tempered / generalized prior density is:
        p_r(Z) ∝ p0(Z) * g(Z, r)
    where log_g_fn(Z, r) = log g(Z, r).

    Parameters
    ----------
    r : float
        Parameter value ("inverse temperature" / deformation parameter).
    log_g_fn : callable
        Function taking (Z, r) returning scalar log g(Z, r). Must be JAX compatible.
    prior : callable
        Base prior distribution p0(Z).
    """
    # Sample Z from base prior p0(Z)
    Z = prior()

    # Add unnormalized log factor
    log_g = log_g_fn(Z, r)
    numpyro.factor("log_g", log_g)
    numpyro.deterministic("_TI_Z", Z)  # allow retrieval with Predictive



def run_thermodynamic_integration(
        rng_key: random.PRNGKey,
        log_g_fn: callable,
        prior: callable,
        r_grid: jax.Array,
        num_samples: int,
        num_warmup: int | None = None,
):
    """Run Thermodynamic Integration for a general log g(Z, r).

    For each r in r_grid we sample from p_r(Z) ∝ p0(Z) g(Z, r) and estimate
    E_r[ ∂/∂r log g(Z, r) ]. We then integrate these estimates over r (trapezoid rule)
    to obtain log c(r) with log c(0) = 0.

    Parameters
    ----------
    rng_key : PRNGKey
        JAX random key.
    log_g_fn : callable
        (Z, r) -> scalar log g(Z, r); must be differentiable w.r.t. r.
    prior : callable
        Base measure p0(Z).
    r_grid : jax.Array
        1D increasing grid of r values (must include 0.0 for reference).
    num_samples : int
        Number of post-warmup samples per r for expectation estimation.
    num_warmup : int | None
        Warmup steps for NUTS. If None, defaults internally to num_samples.

    Returns
    -------
    dlogc_dr_values : jax.Array
        Estimates of E_r[ ∂/∂r log g(Z, r) ] for each r in r_grid.
    log_c_values : jax.Array
        Estimated log c(r) for each r (log normalizing constants).
    """
    if num_warmup is None:
        num_warmup = 10 + int(0.2 * num_samples)

    print(f"Starting generalized TI on {len(r_grid)} grid points...")

    derivative_expectations = []
    keys = random.split(rng_key, len(r_grid))

    # Function to compute ∂/∂r log g(Z, r) for a single Z, r
    def dlogg_dr_single(Z, r):
        return jax.grad(lambda rr: log_g_fn(Z, rr))(r)

    # Vectorized over samples
    v_dlogg_dr = vmap(dlogg_dr_single, in_axes=(0, None))

    kernel = NUTS(model_for_ti, find_heuristic_step_size=True)  #, max_tree_depth=12)
    num_init_samples = int(num_samples * len(r_grid) / 2)
    mcmc_args = dict(num_warmup=num_warmup, num_chains=4,  progress_bar=False)  #, chain_method="vectorized")
    mcmc = MCMC(kernel, num_samples=num_init_samples, **mcmc_args)
    mcmc.run(keys[0], r=r_grid[0], log_g_fn=log_g_fn, prior=prior)
    mcmc_state = mcmc.last_state
    for i, r in enumerate(r_grid):
        print(f"  [TI {i + 1}/{len(r_grid)}] MCMC for r = {r:.3g}")
        mcmc = MCMC(kernel, num_samples=num_samples, **mcmc_args)
        kernel._init_strategy = numpyro.infer.initialization.init_to_value(values=mcmc_state)
        mcmc.run(
            keys[i],
            r=r,
            log_g_fn=log_g_fn,
            prior=prior,
        )

        Z_samples = mcmc.get_samples()["_TI_Z"]  # shape (num_samples, ...)

        # Compute derivative per sample and average
        dlogg_vals = v_dlogg_dr(Z_samples, r)
        expected_deriv = jnp.mean(dlogg_vals)
        derivative_expectations.append(expected_deriv)

        mcmc_state = mcmc.last_state

        print(f"    [expected_deriv] {expected_deriv}")

    print("...General TI MCMC runs complete.")

    dlogc_dr_values = jnp.stack(derivative_expectations)


    # Integrate to get log c(r): log c(r) = ∫_0^r E_r[ ∂_r log g(Z,r) ] dr
    log_c_values = jnp.array(
        cumulative_trapezoid(np.array(dlogc_dr_values), np.array(r_grid), initial=0.0)
    )

    print("...Integration complete.")

    return dlogc_dr_values, log_c_values


def estimate_marginal_log_likelihood_curve(
        base_prior: callable,
        log_g_fn: callable,
        r_grid: jax.Array,
        rng_key: random.PRNGKey = None,
        num_samples: int = 5000,
        num_warmup: int | None = None,
):
    """Pre-compute log c(r) over a grid and return a JAX-differentiable interpolator.

    Parameters
    ----------
    base_prior : callable
        Base prior p0(Z).
    log_g_fn : callable
        (Z, r) -> log g(Z, r) function.
    r_grid : float
        The grid values of r to pre-compute over.
    rng_key : PRNGKey | None
        Random key; if None a fresh one is sampled.
    num_samples : int
        Samples per r (post warmup).
    num_warmup : int | None
        Warmup steps.

    Returns
    -------
    interpolator : RegularGridInterpolator
        Interpolates log c(r) on the grid.
    dlogc_dr_values : jax.Array
        Estimated derivative expectation values.
    log_c_values : jax.Array
        Estimated log c(r) values.
    """
    if rng_key is None:
        rng_key = random.PRNGKey(np.random.randint(2 ** 31, dtype=np.uint32))

    # Bias grid toward small r (square transform) for better resolution near 0
    dlogc_dr_values, log_c_values = run_thermodynamic_integration(
        rng_key, log_g_fn, base_prior, r_grid, num_samples, num_warmup,
    )

    # Order grid and values to make grid ascending

    order = jnp.argsort(r_grid)
    r_grid = r_grid[order]
    log_c_values = log_c_values[order]

    # Ensure values end in 0
    log_c_values -= log_c_values[-1]

    interpolator = RegularGridInterpolator((r_grid,), log_c_values)

    return interpolator, dlogc_dr_values, log_c_values


# --- Example Usage (Toy Problem) ---

def main():
    # --- Define Constants ---
    Z_DIM = 5  # Dimensionality of Z
    R_MAX = 5.0  # Max value of r to pre-compute
    TI_GRID_SIZE = 21  # Number of grid points for TI

    # MCMC settings for TI
    TI_NUM_SAMPLES = 2000  # reduce a bit for quicker demo
    TI_NUM_WARMUP = 1000

    # MCMC settings for final model inference
    FINAL_NUM_SAMPLES = 2000
    FINAL_NUM_WARMUP = 1000

    # --- Define log f(Z) ---
    def log_f(Z):
        """Toy unnormalized log density (GMM with two modes) for demonstration."""
        mu1 = jnp.ones(Z_DIM) * 2.0
        mu2 = jnp.ones(Z_DIM) * -2.0
        log_prob_c1 = dist.Normal(mu1, 1.0).log_prob(Z).sum()
        log_prob_c2 = dist.Normal(mu2, 1.0).log_prob(Z).sum()
        return jnp.logaddexp(log_prob_c1, log_prob_c2) - jnp.log(2.0)

    # General log g(Z, r). For original special case: log g = r * log f.
    def log_g(Z, r):
        return r * log_f(Z)

    # Base prior p0(Z)
    prior = lambda : numpyro.sample("Z", dist.Normal(jnp.zeros(Z_DIM), 1.0).to_event(1))

    r_grid = jnp.linspace(0.0, R_MAX ** 0.5, TI_GRID_SIZE) ** 2

    # === STAGE 1: Pre-compute log c(r) curve ===
    rng_key, data_key, ti_key, mcmc_key = random.split(random.PRNGKey(42), 4)
    log_c_interp, dlogc_dr_values, log_c_values = estimate_marginal_log_likelihood_curve(
        prior, log_g, r_grid, ti_key, TI_NUM_SAMPLES,
    )

    # === Plot TI Results (Sanity / Consistency Checks) ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(r_grid, dlogc_dr_values, 'o-')
    ax1.set_ylabel("E_r[∂_r log g(Z,r)]")
    ax1.set_title("General Thermodynamic Integration Results")
    ax1.grid(True)

    ax2.plot(r_grid, log_c_values, 'o-r')
    ax2.set_xlabel("r (parameter)")
    ax2.set_ylabel("log c(r)")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # === Generate Synthetic Data ===
    N_data = 50
    Z_true_val = 2.0
    sigma_true = 0.5
    y_data = dist.Normal(Z_true_val, sigma_true).sample(data_key, (N_data,))
    print(f"Generated {N_data} data points around y = {Z_true_val:.2f}\n")

    # === Full Model with Unknown r ===
    def full_model(Z_dim, log_c_interpolator, r_max, y_obs=None):
        # Prior for r (bounded by interpolation domain)
        r = numpyro.sample("r", dist.Uniform(0.0, float(r_max)))
        log_c_r = log_c_interpolator(jnp.array([r]))[0]
        # Sample Z from base prior
        Z = numpyro.sample("Z", dist.Normal(jnp.zeros(Z_dim), 1.0).to_event(1))
        # Add generalized prior factor: log g(Z,r) - log c(r)
        numpyro.factor("Z_prior_factor", log_g(Z, r) - log_c_r)
        # Likelihood: observe noisy first component
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        mu_obs = Z[0]
        with numpyro.plate("data", len(y_obs) if y_obs is not None else 1):
            numpyro.sample("obs", dist.Normal(mu_obs, sigma), obs=y_obs)

    full_model_with_interp = partial(
        full_model,
        Z_dim=Z_DIM,
        log_c_interpolator=log_c_interp,
        r_max=R_MAX,
    )

    print("\n--- Running Full Model Inference ---")
    kernel = NUTS(full_model_with_interp)
    mcmc = MCMC(kernel, num_warmup=FINAL_NUM_WARMUP, num_samples=FINAL_NUM_SAMPLES)
    mcmc.run(mcmc_key, y_obs=y_data)
    mcmc.print_summary()

    samples = mcmc.get_samples()
    r_posterior = samples["r"]
    plt.figure(figsize=(10, 4))
    plt.hist(r_posterior, bins=50, density=True, label="Posterior of r")
    mean_r = float(jnp.mean(r_posterior))
    plt.axvline(mean_r, color='red', linestyle='--', label=f"Mean r = {mean_r:.2f}")
    plt.title("Posterior Distribution for 'r'")
    plt.xlabel("r")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()