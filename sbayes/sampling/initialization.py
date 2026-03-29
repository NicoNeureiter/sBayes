import warnings
from functools import partial

from numpyro.infer import SVI, Trace_ELBO, init_to_feasible, init_to_value, MCMC, NUTS, init_to_mean
from numpyro.infer.autoguide import AutoNormal, AutoDelta
from numpyro.infer.util import log_density
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam
from numpyro import handlers
import jax
import jax.numpy as jnp
from numpyro.util import find_stack_level
from tqdm import tqdm


def _sample_has_nans(sample):
    """Check whether any values in the sample dictionary contain NaN."""
    return any(jnp.any(jnp.isnan(v)) for v in sample.values())


def _sample_has_infs(sample):
    """Check whether any values in the sample dictionary contain Inf."""
    return any(jnp.any(jnp.isinf(v)) for v in sample.values())


def _validate_init_sample(model, init_sample, model_args=(), model_kwargs=None, rng_key=None):
    """Validate that the init sample can be used by NUTS without crashing.

    This performs the same checks that NUTS initialization does internally:
    1. Check for NaN/Inf in sample values
    2. Check that log_density is finite
    3. Try the actual NUTS initialization path to catch validation errors
       (e.g. Unit distribution rejects NaN log_factor)
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Check 1: NaN in sample values
    if _sample_has_nans(init_sample):
        return False, "NaN values in the init sample"

    # Check 2: Inf in sample values
    if _sample_has_infs(init_sample):
        return False, "Inf values in the init sample"

    # Check 3: log-density is finite
    try:
        log_prob = log_density(model.get_model, model_args, model_kwargs, init_sample)[0]
        if jnp.isnan(log_prob) or jnp.isinf(log_prob):
            return False, f"non-finite log-probability ({log_prob})"
    except Exception as e:
        return False, f"log_density raised {type(e).__name__}: {e}"

    # Check 4: Try actual NUTS initialization to catch validation errors
    # (e.g. numpyro.factor with NaN log_factor that passes log_density but fails validate_args)
    if rng_key is not None:
        try:
            initialize_model(
                rng_key,
                model.get_model,
                model_args=model_args,
                model_kwargs=model_kwargs,
                init_strategy=init_to_value(values=init_sample),
            )
        except Exception as e:
            return False, f"initialize_model raised {type(e).__name__}: {e}"

    return True, None


def _get_fixed_site_values(model):
    """Get the prior mean values for sites that should be held fixed during SVI.

    Returns a dict mapping site name -> fixed value, only for sites that actually
    exist in the model. Sites are fixed during SVI when gradient-based optimization
    is counterproductive for them (e.g. because they interact pathologically with
    other parameters that start far from their posterior values).
    """
    fixed = {}
    if hasattr(model, 'prior') and hasattr(model.prior, 'geo_prior'):
        geo_cfg = model.prior.geo_prior.config
        if getattr(geo_cfg, 'estimate_rate', False):
            fixed["geoprior_log_scale"] = jnp.log(2 * jnp.array(geo_cfg.rate, dtype=jnp.float32))
    return fixed


def get_svi_init_sample(
    model,
    model_args=(),
    model_kwargs=None,
    rng_key=None,
    svi_steps=100,
    max_retries=3,
    guide_name: str = "AutoDelta",
):
    if model_kwargs is None:
        model_kwargs = {}

    learning_rates = [2e-3, 1e-3, 5e-4]

    # Determine which sites to hold fixed during SVI and their values
    fixed_values = _get_fixed_site_values(model)
    if fixed_values:
        svi_model_fn = handlers.condition(model.get_model, data=fixed_values)
    else:
        svi_model_fn = model.get_model

    heuristic_init = None
    # try:
    #     heuristic_init = find_best_initial_sample(model, rng_key=rng_key)
    # except Exception as e:
    #     warnings.warn(f"Failed to compute heuristic init for SVI: {e}. Falling back to init_to_mean.")

    for attempt in range(max_retries):
        rng_keys = jax.random.split(rng_key, 4)

        lr = learning_rates[min(attempt, len(learning_rates) - 1)]

        if heuristic_init is not None:
            init_loc_fn = init_to_value(values={k: v for k, v in heuristic_init.items() if k not in fixed_values})
        else:
            init_loc_fn = init_to_mean

        if guide_name == "AutoDelta":
            guide = AutoDelta(svi_model_fn, init_loc_fn=init_loc_fn)
        elif guide_name == "AutoNormal":
            guide = AutoNormal(svi_model_fn, init_loc_fn=init_loc_fn)
        else:
            raise ValueError(f"Unknown SVI guide: {guide_name}")
        optimizer = Adam(lr)

        svi = SVI(svi_model_fn, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(rng_keys[1], svi_steps, progress_bar=True, *model_args, **model_kwargs)

        # Check if SVI diverged
        if not jnp.isfinite(svi_result.losses[-1]):
            warnings.warn(
                f"SVI attempt {attempt + 1}/{max_retries} diverged with {guide_name} "
                f"(final loss = {svi_result.losses[-1]}). Retrying with learning rate {lr / 2:.1e}..."
            )
            rng_key = rng_keys[0]  # Use a different key for the next attempt
            continue

        # Return samples from the variational approximation
        init_sample = guide.sample_posterior(rng_keys[2], svi_result.params)

        # Add the fixed site values back into the sample for MCMC initialization
        init_sample.update(fixed_values)

        # Comprehensive validation of the init sample
        valid, reason = _validate_init_sample(
            model, init_sample, model_args, model_kwargs, rng_key=rng_keys[3]
        )
        if not valid:
            warnings.warn(
                f"SVI attempt {attempt + 1}/{max_retries} with {guide_name} failed validation: {reason}. "
                f"Retrying with learning rate {lr / 2:.1e}..."
            )
            rng_key = rng_keys[0]
            continue

        log_prob = log_density(model.get_model, model_args, model_kwargs, init_sample)[0]
        print(f"SVI ({guide_name}) sample has log-prob {log_prob}")
        return init_sample

    # All SVI attempts failed – fall back to heuristic initialization
    warnings.warn(
        f"All {max_retries} SVI attempts failed for guide {guide_name}. Falling back to heuristic initialization."
    )
    return find_best_initial_sample(model, rng_key=rng_key)



def init_by_svi(site=None, svi_steps=30):
    """
    Initialize to the prior median. For priors with no `.sample` method implemented,
    we defer to the :func:`init_to_uniform` strategy.

    :param int num_samples: number of prior points to calculate median.
    """
    if site is None:
        return partial(init_by_svi, svi_steps=svi_steps)

    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and not site["fn"].support.is_discrete
    ):
        if site["value"] is not None:
            warnings.warn(
                f"init_to_median() skipping initialization of site '{site['name']}'"
                " which already stores a value.",
                stacklevel=find_stack_level(),
            )
            return site["value"]

        rng_key = site["kwargs"].get("rng_key")
        sample_shape = site["kwargs"].get("sample_shape")

        guide = AutoNormal(site["fn"], init_loc_fn=init_to_feasible)
        # guide = AutoDelta(site["fn"])
        optimizer = Adam(1e-2)

        svi = SVI(site["fn"], guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(rng_key, svi_steps)
        return guide.sample_posterior(jax.random.PRNGKey(1), svi_result.params)


def find_best_initial_sample(model, rng_key=None, num_samples=50):
    """
    Return the best initialization point based on prior samples with the highest joint log prob.
    """
    model_args = ()
    model_kwargs = {}

    keys = jax.random.split(rng_key, num_samples)
    samples = []
    log_probs = []


    for key in keys:
        sample_dict = model.generate_initial_params(key)
        log_joint, _ = log_density(model.get_model, model_args, model_kwargs, sample_dict)
        samples.append(sample_dict)
        log_probs.append(log_joint)

    best_idx = int(jnp.argmax(jnp.stack(log_probs)))

    print(f"Best sample {best_idx} has log-prob {log_probs[best_idx]}")

    # median_sample = {k: jnp.median(jnp.array([s[k] for s in samples]), axis=0) for k in samples[0]}
    # print(f"Median sample has log-prob {log_density(model.get_model, model_args, model_kwargs, median_sample)[0]}")
    #
    # svi_sample = get_svi_init_sample(model, model_args, model_kwargs, rng_key, svi_steps=200)
    # print(f"SVI sample has log-prob {log_density(model.get_model, model_args, model_kwargs, svi_sample)[0]}")

    return samples[best_idx]


def init_by_annealing(model, model_args, model_kwargs, rng_key, steps_per_temp=1000, temp_schedule=None):
    """Run annealed MCMC to find a good initial state. The annealing slowly decreases the temperature, thereby moving
    through the parameter space in the beginning, but gradually focusing on the high posterior areas later on."""
    if temp_schedule is None:
        temp_schedule = [16, 8, 4, 2, 1, 0.5, 0.25]

    init_sample = find_best_initial_sample(model, rng_key=rng_key)

    mcmc = MCMC(
        sampler=NUTS(model.get_model, init_sample), num_warmup=2, num_samples=steps_per_temp,
        num_chains=1,
        progress_bar=False,
    )
    rng_key, subkey = jax.random.split(rng_key, 2)
    mcmc.warmup(rng_key)
    mcmc_state = mcmc.last_state
    for temp in tqdm(temp_schedule, desc="Initialization by annealing..."):
        mcmc.post_warmup_state = mcmc_state
        rng_key, subkey = jax.random.split(rng_key, 2)
        mcmc.run(rng_key=subkey)
        mcmc_state = mcmc.last_state

    return mcmc_state
