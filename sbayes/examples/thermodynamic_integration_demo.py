# --- Example Usage (Toy Problem) ---

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist

from functools import partial
from numpyro.infer import MCMC, NUTS
from sbayes.model.thermodynamic_integration import estimate_marginal_log_likelihood_curve

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