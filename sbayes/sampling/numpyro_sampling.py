from __future__ import annotations

import jax
import numpy as np
import time

from jax import random
from logging import Logger
from numpyro.infer.hmc import HMCState
from numpyro.infer import MCMC, NUTS, init_to_value
from sbayes.model import Model
from sbayes.sampling.initialization import (
    get_svi_init_sample,
    find_best_initial_sample,
    laplace_inverse_mass_matrix,
)
from sbayes.sampling.loggers import OnlineSampleLogger
from sbayes.util import timeit
from tqdm import tqdm

MAX_TREE_DEPTH = 13
"""Maximum tree depth of the NUTS sampler (at most 2^13 - 1 leapfrog steps per step)."""

MAX_DIVERGENT_SHARE = 0.01
"""Share of divergent steps above which the samples are flagged as unreliable."""


def summarize_steps(diverging: np.ndarray, num_steps: np.ndarray) -> list[str]:
    """Summarize the divergences and tree sizes of the sampling steps.

    Args:
        diverging: whether each step diverged
        num_steps: number of leapfrog steps of each step

    Returns:
        The summary lines, ending with a warning if too many steps diverged.
    """
    n = diverging.size
    n_divergent = int(diverging.sum())
    n_saturated = int(np.sum(num_steps >= 2 ** MAX_TREE_DEPTH - 1))
    lines = [
        f"Divergent steps: {n_divergent} / {n} ({n_divergent / n:.1%})",
        f"Steps at the maximum tree depth: {n_saturated} / {n} ({n_saturated / n:.1%})",
        f"Mean number of leapfrog steps: {num_steps.mean():.0f}",
    ]
    if n_divergent > MAX_DIVERGENT_SHARE * n:
        lines.append(
            "Warning: many steps diverged, so the samples may not represent the posterior."
        )
    return lines


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
    svi_guide: str = "AutoLaplace",
    svi_steps: int = 4000,
    laplace_mass_matrix: bool = True,
    adapt_mass_matrix: bool = True,
    logger: Logger | None = None,
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
        laplace_mass_matrix: whether to start the mass-matrix adaptation from the
            curvature at the SVI mode (only with the "SVI" init strategy)
        adapt_mass_matrix: whether to adapt the mass matrix during warm-up. If False,
            the initial one is kept for the whole run.
        logger: where to write progress messages (printed if None)

    Returns:
        The samples of this run, read back from the results file.
    """
    # TODO: re-reading a previous run's samples without sampling was removed here.
    #   If reinstated, it belongs in loggers.py as a standalone function.

    log = logger.info if logger is not None else print

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

        inverse_mass_matrix = None
        if laplace_mass_matrix and init_strategy == "SVI":
            t_start = time.time()
            rng_key, subkey = random.split(rng_key, 2)
            inverse_mass_matrix = laplace_inverse_mass_matrix(model, initial_sample, subkey)
            log(
                f"Initial inverse mass matrix from the curvature at the SVI mode "
                f"({inverse_mass_matrix.shape[0]} parameters, {time.time() - t_start:.1f}s)."
            )

        if not adapt_mass_matrix and inverse_mass_matrix is None:
            log(
                "Warning: the mass matrix is not adapted and not initialized from the "
                "SVI mode, so NUTS uses the identity matrix for the whole run."
            )

        kernel = NUTS(
            model.get_model,
            init_strategy=init_to_value(values=initial_sample),
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_mass_matrix=adapt_mass_matrix,
            find_heuristic_step_size=True,
            max_tree_depth=MAX_TREE_DEPTH,
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
        # Show warm-up progress even when the chunked sampling below shows only a
        # per-chunk progress bar
        mcmc.progress_bar = True
        mcmc.warmup(rng_key=subkey)
        mcmc.progress_bar = not split_runs
        mcmc_state = mcmc.post_warmup_state
        step_size = float(np.mean(mcmc_state.adapt_state.step_size))
        log(f"NUTS step size after warm-up: {step_size:.3g}")
    else:
        mcmc.post_warmup_state = mcmc_state = init_state

    # Per-step diagnostics of the kept (thinned) steps, summarized at the end
    diverging, num_steps = [], []

    def run_chunk(state, key: jax.Array) -> tuple[HMCState, int]:
        """Sample one chunk, write it to disk and return the new state and its size."""
        mcmc.post_warmup_state = state
        mcmc.run(rng_key=key, extra_fields=("potential_energy", "diverging", "num_steps"))

        samples = mcmc.get_samples(group_by_chain=True)
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
        diverging.append(np.asarray(extra_fields["diverging"]).ravel())
        num_steps.append(np.asarray(extra_fields["num_steps"]).ravel())
        samples["potential_energy"] = extra_fields["potential_energy"]

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

    for line in summarize_steps(np.concatenate(diverging), np.concatenate(num_steps)):
        log(line)
    return sample_logger.read_samples()

