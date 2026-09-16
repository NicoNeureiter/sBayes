from __future__ import annotations

import jax
import jax.numpy as jnp
import warnings

from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO, init_to_value, init_to_mean
from numpyro.infer.autoguide import AutoNormal, AutoDelta
from numpyro.infer.util import log_density
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam
from sbayes.util import FLOAT_TYPE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sbayes.model import Model


def _non_finite_sites(sample: dict) -> list[str]:
    """Return the names of the sites whose values are not all finite."""
    return [
        name for name, value in sample.items()
        if not jnp.isfinite(jnp.asarray(value)).all()
    ]

def _validate_init_sample(
    model: Model,
    init_sample: dict,
    rng_key: jax.Array | None = None,
) -> tuple[bool, str | None]:
    """Check that an init sample can be used to start NUTS.

    Runs the same checks as NUTS initialization: the values must be finite, the joint
    log-density must be finite, and the sample must survive `initialize_model`, which
    also validates the gradient.

    Args:
        model: the model the sample initializes
        init_sample: the site values to start from
        rng_key: random key for the `initialize_model` check; skipped if not given

    Returns:
        Whether the sample is usable, and the reason if it is not.
    """
    bad_sites = _non_finite_sites(init_sample)
    if bad_sites:
        return False, f"non-finite values at {bad_sites}"

    try:
        log_prob = log_density(model.get_model, (), {}, init_sample)[0]
    except Exception as e:
        return False, f"log_density raised {type(e).__name__}: {e}"

    if not jnp.isfinite(log_prob):
        return False, f"non-finite log-probability ({log_prob})"

    if rng_key is not None:
        # Catches what the log-density alone does not, in particular a non-finite
        # gradient and distributions that reject their arguments
        try:
            initialize_model(
                rng_key,
                model.get_model,
                init_strategy=init_to_value(values=init_sample),
            )
        except Exception as e:
            return False, f"initialize_model raised {type(e).__name__}: {e}"

    return True, None


def _get_fixed_site_values(model: Model) -> dict:
    """Return the sites to hold fixed during SVI, and the values to fix them at.

    The geo-prior rate is held fixed because optimizing it jointly with the cluster
    assignments pulls it towards degenerate values: a small rate makes any clustering
    look good, so SVI shrinks it instead of finding clusters.

    Args:
        model: the model whose sites are to be fixed

    Returns:
        A mapping of site name to fixed value, empty if nothing needs fixing.
    """
    geo_config = model.prior.geo_prior.config
    if not geo_config.estimate_rate:
        return {}

    # Twice the configured rate, so the geo-prior is weak enough not to dominate the
    # SVI objective while the cluster assignments are still far from the posterior
    return {"geoprior_log_scale": jnp.log(2 * jnp.array(geo_config.rate, dtype=FLOAT_TYPE))}


def get_svi_init_sample(
    model: Model,
    rng_key: jax.Array,
    svi_steps: int = 100,
    max_retries: int = 3,
    guide_name: str = "AutoDelta",
) -> dict:
    """Find a starting point for MCMC by fitting a variational approximation.

    Runs SVI and draws a sample from the fitted guide. If the optimization diverges or
    the resulting sample cannot be used to start NUTS, the fit is retried with a smaller
    learning rate. After `max_retries` failures, the heuristic initialization is used
    instead.

    Args:
        model: the model to initialize
        rng_key: JAX random key
        svi_steps: number of optimization steps per attempt
        max_retries: number of attempts before falling back to the heuristic
        guide_name: the guide family, either "AutoDelta" or "AutoNormal"

    Returns:
        The site values to start the sampler from.
    """
    learning_rates = [2e-3, 1e-3, 5e-4]

    # Sites that are held fixed during the optimization
    fixed_values = _get_fixed_site_values(model)
    svi_model_fn = (
        handlers.condition(model.get_model, data=fixed_values)
        if fixed_values else model.get_model
    )

    for attempt in range(max_retries):
        rng_keys = jax.random.split(rng_key, 4)
        learning_rate = learning_rates[min(attempt, len(learning_rates) - 1)]

        if guide_name == "AutoDelta":
            guide = AutoDelta(svi_model_fn, init_loc_fn=init_to_mean)
        elif guide_name == "AutoNormal":
            guide = AutoNormal(svi_model_fn, init_loc_fn=init_to_mean)
        else:
            raise ValueError(
                f"Unknown SVI guide `{guide_name}` (choose from: AutoDelta, AutoNormal)."
            )

        svi = SVI(svi_model_fn, guide, Adam(learning_rate), loss=Trace_ELBO())
        svi_result = svi.run(rng_keys[1], svi_steps, progress_bar=True)

        if not jnp.isfinite(svi_result.losses[-1]):
            reason = f"the optimization diverged (final loss {svi_result.losses[-1]})"
        else:
            # Draw a sample from the fitted guide and put the fixed sites back in
            init_sample = guide.sample_posterior(rng_keys[2], svi_result.params)
            init_sample.update(fixed_values)

            valid, reason = _validate_init_sample(
                model, init_sample, rng_key=rng_keys[3]
            )
            if valid:
                return init_sample

        warnings.warn(
            f"SVI initialization attempt {attempt + 1}/{max_retries} with {guide_name} "
            f"failed: {reason}."
        )
        rng_key = rng_keys[0]

    warnings.warn(
        f"All {max_retries} SVI initialization attempts failed with {guide_name}. "
        f"Falling back to heuristic initialization."
    )
    return find_best_initial_sample(model, rng_key=rng_key)


def find_best_initial_sample(
    model: Model,
    rng_key: jax.Array,
    num_samples: int = 50,
) -> dict:
    """Find a starting point for MCMC among a set of prior samples.

    Draws `num_samples` initial samples and returns the one with the highest joint
    log-density. This is the "heuristic" initialization strategy, and the fallback when
    the SVI initialization fails.

    Args:
        model: the model to initialize
        rng_key: JAX random key
        num_samples: number of candidate samples to draw

    Returns:
        The candidate with the highest joint log-density.

    Raises:
        RuntimeError: if the best candidate cannot be used to start the sampler.
    """
    rng_key, validation_key = jax.random.split(rng_key, 2)

    samples = []
    log_probs = []

    for key in jax.random.split(rng_key, num_samples):
        sample = model.generate_initial_params(key)
        log_joint, _ = log_density(model.get_model, (), {}, sample)
        samples.append(sample)
        log_probs.append(log_joint)

    best_sample = samples[int(jnp.argmax(jnp.stack(log_probs)))]

    valid, reason = _validate_init_sample(model, best_sample, rng_key=validation_key)
    if not valid:
        raise RuntimeError(
            f"Heuristic initialization failed: the best of {num_samples} candidate "
            f"samples cannot be used to start the sampler ({reason})."
        )

    return best_sample