import jax

from jax import random
from numpyro.infer.hmc import HMCState
from numpyro.infer import MCMC, NUTS, init_to_value
from sbayes.model import Model
from sbayes.sampling.initialization import get_svi_init_sample, find_best_initial_sample
from sbayes.sampling.loggers import OnlineSampleLogger
from sbayes.util import timeit
from tqdm import tqdm


@timeit('s')
def sample_nuts(
    model: Model,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key: jax.Array,
    write_interval: int,
    sample_logger: OnlineSampleLogger,
    thinning: int = 1,
    init_state: HMCState | None = None,
    init_strategy: str = "SVI",
    svi_guide: str = "AutoDelta",
    svi_steps: int = 4000,
) -> dict:
    """Sample the model with the NUTS sampler, writing samples to disk as they arrive.

    The sampling is split into chunks of `write_interval` steps, so that samples are
    written to the results file during the run rather than only at the end. Each chunk
    resumes from the sampler state of the previous one.

    Args:
        model: the model to sample from
        num_warmup: number of warm-up (adaptation) steps
        num_samples: number of sampling steps, before thinning
        num_chains: number of chains to run
        rng_key: JAX random key
        write_interval: number of steps between writes to the results file
        thinning: keep only every `thinning`-th sample
        init_state: sampler state of a previous run to resume from. If given, warm-up is
            skipped, since the state carries its own adaptation.
        init_strategy: how to find a starting point when not resuming, either "SVI" or
            "heuristic"
        sample_logger: the logger writing the samples to disk
        svi_guide: the guide family used by the "SVI" init strategy
        svi_steps: number of SVI steps used by the "SVI" init strategy

    Returns:
        The samples of this run, read back from the results file.
    """
    # TODO: re-reading a previous run's samples without sampling was removed here.
    #   If reinstated, it belongs in loggers.py as a standalone function.

    if init_state is None:
        # Find a good starting point for the sampler
        if init_strategy == "SVI":
            initial_sample = get_svi_init_sample(
                model, rng_key=rng_key, svi_steps=svi_steps, guide_name=svi_guide
            )
        elif init_strategy == "heuristic":
            initial_sample = find_best_initial_sample(model, rng_key=rng_key)
        else:
            raise ValueError(
                f"Unknown initialization strategy `{init_strategy}` "
                f"(choose from: SVI, heuristic)."
            )

        kernel = NUTS(
            model.get_model,
            init_strategy=init_to_value(values=initial_sample),
            find_heuristic_step_size=True,
            max_tree_depth=13,
            target_accept_prob=0.7,
        )
    else:
        # Resuming: the sampler state carries its own adaptation, so neither an
        # initialization strategy nor warm-up is needed
        kernel = NUTS(model.get_model)

    num_writes = num_samples // write_interval
    split_runs = num_writes >= 2

    mcmc = MCMC(
        sampler=kernel,
        num_warmup=num_warmup,
        num_samples=write_interval if split_runs else num_samples,
        num_chains=num_chains,
        thinning=thinning,
        progress_bar=not split_runs,
        chain_method="vectorized",
    )

    if init_state is None:
        rng_key, subkey = random.split(rng_key, 2)
        mcmc.warmup(rng_key=subkey)
        mcmc_state = mcmc.post_warmup_state
    else:
        mcmc.post_warmup_state = mcmc_state = init_state

    def run_chunk(state, key: jax.Array) -> tuple[HMCState, int]:
        """Sample one chunk, write it to disk and return the new state and its size."""
        mcmc.post_warmup_state = state
        mcmc.run(rng_key=key, extra_fields=("potential_energy",))

        samples = mcmc.get_samples(group_by_chain=True)
        samples["potential_energy"] = mcmc.get_extra_fields(
            group_by_chain=True
        )["potential_energy"]

        sample_logger.write_sample(samples)
        sample_logger.dump_state(mcmc.last_state)

        return mcmc.last_state, samples["potential_energy"].shape[1]

    if split_runs:
        num_samples_done = 0
        for _ in tqdm(range(num_writes)):
            rng_key, subkey = random.split(rng_key, 2)
            mcmc_state, chunk_size = run_chunk(mcmc_state, subkey)
            num_samples_done += chunk_size

        # `num_samples` may not be a multiple of `write_interval`, so sample the rest
        if num_samples_done < num_samples // thinning:
            mcmc.num_samples = num_samples - num_samples_done * thinning
            rng_key, subkey = random.split(rng_key, 2)
            run_chunk(mcmc_state, subkey)
    else:
        run_chunk(mcmc_state, rng_key)

    return sample_logger.read_samples()

