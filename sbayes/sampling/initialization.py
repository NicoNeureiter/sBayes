from __future__ import annotations

import jax
import jax.numpy as jnp
import warnings

from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO, init_to_value, init_to_uniform
from numpyro.infer.autoguide import AutoDelta, AutoLaplaceApproximation, AutoNormal
from numpyro.infer.util import log_density
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam
from jax.flatten_util import ravel_pytree
from sbayes.util import FLOAT_TYPE
from logging import Logger
from typing import TYPE_CHECKING, Callable

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


_GUIDE_CLASSES = {
    "AutoLaplace": AutoLaplaceApproximation,
    "AutoDelta": AutoDelta,
    "AutoNormal": AutoNormal,
}


def _fit_svi(
    model: Model,
    model_fn: Callable,
    fixed_values: dict,
    rng_key: jax.Array,
    svi_steps: int,
    max_retries: int,
    guide_name: str,
    log: Callable[[str], None],
) -> dict | None:
    """Fit one SVI guide from a random starting point and return its median.

    Retries with smaller learning rates; returns None if all `max_retries` fits fail.

    Args:
        model: the model to initialize, used to validate the result
        model_fn: the model function to fit, with the fixed sites conditioned on
        fixed_values: the fixed sites, put back into the result
        rng_key: JAX random key
        svi_steps: number of optimization steps per fit
        max_retries: number of fits before giving up
        guide_name: "AutoLaplace", "AutoDelta" or "AutoNormal"
        log: where to write progress messages
    """
    learning_rates = [2e-3, 1e-3, 5e-4]

    for retry in range(max_retries):
        rng_key, svi_key, validation_key = jax.random.split(rng_key, 3)
        learning_rate = learning_rates[min(retry, len(learning_rates) - 1)]

        guide = _GUIDE_CLASSES[guide_name](model_fn, init_loc_fn=init_to_uniform)
        svi = SVI(model_fn, guide, Adam(learning_rate), loss=Trace_ELBO())
        svi_result = svi.run(svi_key, svi_steps, progress_bar=True)

        if not jnp.isfinite(svi_result.losses[-1]):
            reason = f"the optimization diverged (final loss {svi_result.losses[-1]})"
        else:
            # Start from the guide's median (not a random draw) and put the fixed
            # sites back in
            init_sample = dict(guide.median(svi_result.params))
            init_sample.update(fixed_values)

            valid, reason = _validate_init_sample(model, init_sample, rng_key=validation_key)
            if valid:
                return init_sample

        log(f"SVI fit {retry + 1}/{max_retries} (learning rate {learning_rate}) failed: {reason}.")

    return None


def get_svi_init_sample(
    model: Model,
    rng_key: jax.Array,
    svi_steps: int = 100,
    n_attempts: int = 1,
    max_retries: int = 3,
    guide_name: str = "AutoLaplace",
    logger: Logger | None = None,
) -> dict:
    """Find a starting point for MCMC by fitting SVI from several starting points.

    Runs SVI `n_attempts` times and returns the mode with the highest local evidence
    (see `log_local_evidence`). Falls back to the heuristic initialization if no attempt
    succeeds.

    AutoLaplace finds the mode in unconstrained space, where NUTS samples. AutoDelta's
    mode in constrained space can lie on the simplex boundary, where NUTS cannot start.

    Args:
        model: the model to initialize
        rng_key: JAX random key
        svi_steps: number of optimization steps per fit
        n_attempts: number of SVI runs from different random starting points
        max_retries: fits per attempt (with smaller learning rates) before giving up on it
        guide_name: "AutoLaplace", "AutoDelta" or "AutoNormal"
        logger: where to write progress messages (printed if None)

    Returns:
        The site values to start the sampler from.
    """
    if guide_name not in _GUIDE_CLASSES:
        raise ValueError(
            f"Unknown SVI guide `{guide_name}` (choose from: {', '.join(_GUIDE_CLASSES)})."
        )
    log = logger.info if logger is not None else print

    # Sites that are held fixed during the optimization
    fixed_values = _get_fixed_site_values(model)
    svi_model_fn = (
        handlers.condition(model.get_model, data=fixed_values)
        if fixed_values else model.get_model
    )

    best_sample, best_evidence = None, -jnp.inf
    for attempt, key in enumerate(jax.random.split(rng_key, n_attempts)):
        fit_key, evidence_key = jax.random.split(key)
        init_sample = _fit_svi(
            model, svi_model_fn, fixed_values, fit_key,
            svi_steps=svi_steps, max_retries=max_retries, guide_name=guide_name, log=log,
        )
        if init_sample is None:
            log(f"SVI attempt {attempt + 1}/{n_attempts} failed.")
            continue
        if n_attempts == 1:
            return init_sample

        # Score the mode of the model the optimization saw, with the fixed sites
        # conditioned on, so the scores of all attempts are comparable
        evidence = log_local_evidence(svi_model_fn, init_sample, evidence_key)
        log(f"SVI attempt {attempt + 1}/{n_attempts}: log local evidence {evidence:.2f}.")
        if evidence > best_evidence:
            best_sample, best_evidence = init_sample, evidence

    if best_sample is not None:
        return best_sample

    warnings.warn(
        f"All {n_attempts} SVI initialization attempts failed with {guide_name}. "
        f"Falling back to heuristic initialization."
    )
    return find_best_initial_sample(model, rng_key=rng_key)


def log_local_evidence(
    model_fn: Callable,
    init_sample: dict,
    rng_key: jax.Array,
    num_samples: int = 256,
) -> float:
    """Estimate the log evidence of the posterior mode at `init_sample`.

    Importance sampling with a diagonal Laplace proposal around the mode, so the result
    is roughly the log posterior mass of this mode. Meant to rule out clearly worse
    modes, not to rank close ones: it is a lower bound in expectation.

    Args:
        model_fn: the model function
        init_sample: the site values at the mode; conditioned sites are ignored
        rng_key: JAX random key
        num_samples: number of importance samples

    Returns:
        The log evidence estimate.
    """
    init_key, sample_key = jax.random.split(rng_key)
    flat_z, potential, variance = diagonal_laplace(model_fn, init_sample, init_key)
    std = jnp.sqrt(variance)

    eps = jax.random.normal(sample_key, (num_samples, flat_z.shape[0]), dtype=flat_z.dtype)
    # log q(z) of a diagonal Gaussian, at z = mode + std * eps
    log_q = jax.scipy.stats.norm.logpdf(eps).sum(axis=1) - jnp.log(std).sum()
    # Evaluated in batches, to bound the memory of large models
    log_joint = -jax.lax.map(
        jax.jit(lambda e: potential(flat_z + std * e)), eps, batch_size=32
    )

    # A NaN weight (a sample outside the model's support) would make the sum NaN
    log_w = jnp.where(jnp.isnan(log_joint), -jnp.inf, log_joint - log_q)
    return float(jax.scipy.special.logsumexp(log_w) - jnp.log(num_samples))


def diagonal_laplace(
    model_fn: Callable,
    init_sample: dict,
    rng_key: jax.Array,
    chunk_size: int = 64,
    min_curvature: float = 1e-6,
) -> tuple[jnp.ndarray, Callable, jnp.ndarray]:
    """Diagonal Laplace approximation at `init_sample`, in NUTS's unconstrained space.

    The variances are 1 / H_ii of the potential energy, computed with one Hessian-vector
    product per parameter (in chunks), so the full Hessian is never stored. Directions
    with non-positive or non-finite curvature get variance 1.

    Args:
        model_fn: the model function
        init_sample: the site values to evaluate the curvature at
        rng_key: JAX random key, only used to set up the model
        chunk_size: number of Hessian-vector products evaluated at once
        min_curvature: curvatures below this get variance 1

    Returns:
        The flat unconstrained values (in NUTS's site order), the potential energy as a
        function of them, and the variances.
    """
    model_info = initialize_model(
        rng_key, model_fn, init_strategy=init_to_value(values=init_sample)
    )
    # NUTS ravels the same unconstrained parameters in the same (sorted) site order
    flat_z, unravel = ravel_pytree(model_info.param_info.z)
    n_params = flat_z.shape[0]

    def potential(flat: jnp.ndarray) -> jnp.ndarray:
        return model_info.potential_fn(unravel(flat))

    grad_fn = jax.grad(potential)

    @jax.jit
    def hessian_diagonal_chunk(indices: jnp.ndarray) -> jnp.ndarray:
        basis = jax.nn.one_hot(indices, n_params, dtype=flat_z.dtype)
        hvp = jax.vmap(lambda v: jax.jvp(grad_fn, (flat_z,), (v,))[1])(basis)
        return hvp[jnp.arange(indices.shape[0]), indices]

    chunk_size = min(chunk_size, n_params)
    diagonal = []
    for start in range(0, n_params, chunk_size):
        # Pad the last chunk to the full size, so that it reuses the compiled function
        indices = jnp.minimum(jnp.arange(start, start + chunk_size), n_params - 1)
        diagonal.append(hessian_diagonal_chunk(indices)[: n_params - start])
    curvature = jnp.concatenate(diagonal)

    valid = jnp.isfinite(curvature) & (curvature > min_curvature)
    variance = jnp.where(valid, 1.0 / jnp.where(valid, curvature, 1.0), 1.0)
    return flat_z, potential, variance


def laplace_inverse_mass_matrix(
    model: Model,
    init_sample: dict,
    rng_key: jax.Array,
    chunk_size: int = 64,
    min_curvature: float = 1e-6,
) -> jnp.ndarray:
    """Diagonal inverse mass matrix from the curvature at `init_sample` (the SVI mode).

    Gives NUTS realistic per-parameter scales to start the mass-matrix adaptation from.
    See `diagonal_laplace`.

    Args:
        model: the model to sample from
        init_sample: the site values to evaluate the curvature at
        rng_key: JAX random key, only used to set up the model
        chunk_size: number of Hessian-vector products evaluated at once
        min_curvature: curvatures below this keep the default value of 1

    Returns:
        The diagonal inverse mass matrix, flat in NUTS's site order.
    """
    _, _, variance = diagonal_laplace(
        model.get_model, init_sample, rng_key, chunk_size, min_curvature
    )
    return variance


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