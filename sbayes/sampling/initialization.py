import warnings
from functools import partial

from numpyro.infer import SVI, Trace_ELBO, init_to_feasible
from numpyro.infer.autoguide import AutoNormal, AutoDelta
from numpyro.infer.util import log_density, unconstrain_fn, transform_fn
from numpyro.optim import Adam
from numpyro.handlers import seed, trace
import jax
import jax.numpy as jnp
from numpyro.util import find_stack_level


def get_svi_init_sample(model, model_args=(), model_kwargs=None, rng_key=None, svi_steps=100, num_chains=1):
    if model_kwargs is None:
        model_kwargs = {}

    # guide = AutoNormal(model.get_model, init_loc_fn=init_to_feasible)
    guide = AutoDelta(model.get_model)
    optimizer = Adam(3e-3)

    svi = SVI(model.get_model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, svi_steps, *model_args, **model_kwargs)

    # Return samples from the variational approximation
    return guide.sample_posterior(jax.random.PRNGKey(1), svi_result.params)

    # def init_fn(site):
    #     if (site["type"] == "sample"
    #         and not site["is_observed"]
    #         and not site["fn"].support.is_discrete
    #     ):
    #         print(site)
    #         exit()
    #         return samples[site["name"]]
    #     # if site["name"] in samples:
    #     #     return samples[site["name"]]
    #     # raise ValueError(f"Unknown site name: {site['name']}")
    #
    # return init_fn

    # samples = []
    # for i in range(num_chains):
    #     s = guide.sample_posterior(jax.random.PRNGKey(1), svi_result.params)
    #     s = unconstrain_fn(model.get_model, (), {}, s)
    #     samples.append(s)
    #
    # # stack samples into a single dictionary
    # samples = {k: jnp.stack([s[k] for s in samples]) for k in samples[0]}
    #
    # # samples = guide.sample_posterior(jax.random.PRNGKey(1), svi_result.params)
    # # for key, value in samples.items():
    # #     samples[key] = jnp.broadcast_to(value, (num_chains,) + value.shape)
    #
    # return samples


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


def find_best_initial_sample(model, model_args=(), model_kwargs=None, rng_key=None, num_samples=100):
    """
    Return the best initialization point based on prior samples with highest joint log prob.
    """
    if model_kwargs is None:
        model_kwargs = {}

    keys = jax.random.split(rng_key, num_samples)
    samples = []
    log_probs = []


    for key in keys:
        # prior_trace = trace(seed(model, key)).get_trace(*model_args, **model_kwargs)
        #
        # sample_dict = {
        #     name: site["value"]
        #     for name, site in prior_trace.items()
        #     if site["type"] == "sample" and not site["is_observed"]
        # }
        sample_dict = model.generate_initial_params(key)

        log_joint, _ = log_density(model.get_model, model_args, model_kwargs, sample_dict)
        # print(log_joint)

        samples.append(sample_dict)
        log_probs.append(log_joint)

    best_idx = int(jnp.argmax(jnp.stack(log_probs)))

    # print(f"Best sample {best_idx} has log-prob {log_probs[best_idx]}")
    #
    # median_sample = {k: jnp.median(jnp.array([s[k] for s in samples]), axis=0) for k in samples[0]}
    # print(f"Median sample has log-prob {log_density(model.get_model, model_args, model_kwargs, median_sample)[0]}")
    #
    # svi_sample = get_svi_init_sample(model, model_args, model_kwargs, rng_key, svi_steps=200)
    # print(f"SVI sample has log-prob {log_density(model.get_model, model_args, model_kwargs, svi_sample)[0]}")

    return samples[best_idx]

