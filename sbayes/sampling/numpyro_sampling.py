import jax.numpy as jnp
import jax
from jax import random, lax
from jax import nn
from jax.example_libraries.optimizers import inverse_time_decay
from jax.nn import softmax

import numpyro
from numpyro.distributions.transforms import StickBreakingTransform
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive, MixedHMC, DiscreteHMCGibbs
import numpyro.distributions as dist
from numpyro.distributions import transforms, constraints
# from numpyro.contrib.tfp.mcmc import ReplicaExchangeMC
# from numpyro.contrib.nested_sampling import NestedSampler  # Changes quite a few jax settings (e.g. x64)
from numpyro import handlers
from numpyro.infer import autoguide
import matplotlib.pyplot as plt
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.util import initialize_model
from numpyro.optim import Adam

from sbayes.model import Model

# from tensorflow_probability.substrates import jax as tfp

numpyro.set_host_device_count(4)
# jax.config.update("jax_traceback_filtering", "off")


def get_model_dims(model, rng_key):
    init_params, _, _, _ = initialize_model(rng_key, model)
    dim = sum(p.size for p in init_params.z.values())
    return dim

def get_model_shapes(model, rng_key):
    init_params, _, _, _ = initialize_model(rng_key, model)
    return {k: v.size for k, v in init_params.z.items()}

def sample_nuts(
    model,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key: random.PRNGKey,
    thinning: int = 1,
    init_params: dict = None,
):
    # # # Sample the model parameters and latent variables using NUTS
    # dim = get_model_dims(model, rng_key)
    # # model_shapes = get_model_shapes(model, rng_key)
    # # model_params = tuple(model_shapes.keys())
    # # print(dim)
    # # exit()
    # #
    # # inv_mass_matrix = {model_params: jnp.full(dim, 0.01, dtype=jnp.float32)}
    # # kernel = NUTS(
    # #     model,
    # #     inverse_mass_matrix=inv_mass_matrix,
    # #     adapt_mass_matrix=False,
    # #     dense_mass=False,
    # # )

    inner_kernel = NUTS(model)
    # kernel = MixedHMC(inner_kernel=inner_kernel, num_discrete_updates=10)
    # kernel = DiscreteHMCGibbs(inner_kernel=inner_kernel)
    kernel = inner_kernel

    mcmc = MCMC(
        sampler=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
    )
    mcmc.run(
        rng_key,
        init_params=init_params,
        extra_fields=("potential_energy",),
        # extra_fields=("potential_energy", "adapt_state.step_size", "adapt_state.inverse_mass_matrix"),
    )

    show_inference_summary = False
    if show_inference_summary:
        mcmc.print_summary()
        # print(mcmc.get_extra_fields())

    return mcmc, mcmc.get_samples(group_by_chain=False)


def get_manual_guide(model: Model):
    """
    Create a manual guide (variational approximation) for the sBayes model.
    """
    n_clusters = model.n_clusters
    n_objects = model.shapes.n_sites
    n_features = model.shapes.n_features

    features = model.data.features.values
    confounders = list(model.data.confounders.values())
    counts_by_conf = [
        jnp.array([
            jnp.sum(features[grps, :, :], axis=0)
            for grps in conf.group_assignment
        ] + [jnp.zeros((n_features, model.shapes.n_states))])  # Add a dummy group
        for conf in confounders
    ]

    def guide(*args, **kwargs):
        # Guide for `z` (cluster assignments)
        # # z = add_logistic_normal_distribution(n_clusters + 1, n_objects, "z")
        # z = add_logistic_normal_distribution(n_objects, n_clusters + 1, "z")
        # # z_posterior_conc = numpyro.param("z_posterior_conc", jnp.ones((n_objects, n_clusters + 1)), constraint=constraints.positive)
        # # numpyro.sample("z", dist.Dirichlet(z_posterior_conc))

        # Parameters for the logistic normal
        # z_mean = numpyro.param("z_mean", jnp.zeros((n_clusters, n_objects)))
        z_mean = numpyro.param("z_mean", 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_clusters, n_objects)))
        z_cov = numpyro.param("z_cov", jnp.eye(n_objects), constraints=constraints.positive_semidefinite)
        z_cov += 1e-6 * jnp.eye(n_objects)  # Add a small diagonal noise term for stability

        # Sample from the normal distribution with object-wise correlation
        z_logit = numpyro.sample("z_logit", dist.MultivariateNormal(z_mean, z_cov),
                                 infer={'is_auxiliary': True})

        # Apply the softmax transform
        z_logit_scale = numpyro.param("z_logit_scale", 0.1, constraint=constraints.positive)
        z = numpyro.sample("z", dist.TransformedDistribution(
            dist.Normal(z_logit.T, z_logit_scale),
            transforms=[transforms.StickBreakingTransform()],
            # transforms=[SoftmaxTransform()],
        ))

        # z = numpyro.sample("z", dist.TransformedDistribution(
        #     dist.MultivariateNormal(z_mean, z_cov),
        #     transforms=[Transpose(), SoftmaxTransform()],
        # ))

        print(z.shape)

        n_states = model.shapes.n_states
        # cluster_eff_logit_mean = fnn_two_layers(z.T, D_H = 5 * n_features, D_Y=n_features * n_states)[:-1, :].reshape((n_clusters, n_features, n_states))
        # cluster_eff_logit_mean = numpyro.param(f"cluster_eff_mean", jnp.zeros((n_clusters, n_features, n_states-1)))
        cluster_eff_logit_mean = numpyro.param(f"cluster_eff_mean", 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n_clusters, n_features, n_states-1)))
        cluster_eff_logit_scale = numpyro.param(f"cluster_eff_scale", jnp.ones((n_clusters, n_features, n_states-1)), constraint=constraints.positive)
        numpyro.sample("cluster_effect", dist.TransformedDistribution(
            dist.Normal(cluster_eff_logit_mean, cluster_eff_logit_scale),
            transforms=[transforms.StickBreakingTransform()],
            # transforms=[SoftmaxTransform()],
        ))

        # cluster_effect_conc = numpyro.param("cluster_effect_conc", jnp.ones((n_clusters, n_features, n_states)), constraint=constraints.positive)
        # print('cluster_effect_conc', cluster_effect_conc.shape)

        # cluster_effect_conc = numpyro.param("cluster_effect_conc", jnp.repeat(model.clust_eff_prior_conc[None,...], n_clusters, axis=0), constraint=constraints.positive)
        # numpyro.sample("cluster_effect", dist.Dirichlet(cluster_effect_conc))

        # Guide for confounding effects
        for i_c in range(1, model.shapes.n_confounders + 1):
            n_groups = len(model.group_names[i_c])
            with numpyro.plate(f"plate_groups_{i_c}", n_groups + 1, dim=-2):
                with numpyro.plate(f"plate_features_{i_c}", model.shapes.n_features, dim=-1):
                    conf_effect_conc = numpyro.param(f"conf_effect_conc_{i_c - 1}", model.conf_eff_prior_conc[i_c - 1] + counts_by_conf[i_c - 1] / model.shapes.n_components,
                                                     constraint=constraints.positive)
                    numpyro.sample(f"conf_eff_{i_c - 1}", dist.Dirichlet(conf_effect_conc))

        # Guide for weights (`w`)
        w_posterior_conc = numpyro.param("w_posterior_conc", model.w_prior_conc,
                                         constraint=constraints.positive)
        numpyro.sample("w", dist.Gamma(w_posterior_conc, 1))
    return guide


def sample_svi(
    model: callable,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key: random.PRNGKey,
    guide_family: str = "AutoNormal",
    # guide_family: str = "Manual",
    DIAS_K: int = 8,
    thinning: int = 1,
    show_inference_summary: bool = True,
    guide = None,
):
    # if guide is not None:
    #     guide_family = "Manual"

    # Choose the guide family for SVI
    if guide_family == "AutoNormal":
        guide = autoguide.AutoNormal(model)
        step_size = 0.01
    elif guide_family == "AutoDAIS":
        guide = autoguide.AutoDAIS(model, K=DIAS_K, eta_init=0.02, eta_max=0.5)
        step_size = 0.005
    elif guide_family == "Manual":
        step_size = 0.002
    else:
        raise ValueError(f"Unknown guide family `{guide_family}`")

    # Define ADAM optimizer
    optimizer = numpyro.optim.ClippedAdam(
        step_size=step_size,
        # step_size=inverse_time_decay(step_size,
        #                              decay_rate=1.0,
        #                              decay_steps=2_000),
        clip_norm=10.0,
    )


    # Run SVI
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(rng_key, 10_000)
    params = svi_result.params

    if show_inference_summary:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(svi_result.losses[10:], label=f"step_size={step_size}")
        ax.set_title("ELBO loss", fontsize=18, fontweight="bold")
        plt.legend()
        plt.show()

    # get posterior samples
    predictive = Predictive(model, guide=guide, params=params, num_samples=num_samples,
                            return_sites=["z"] + list(get_model_shapes(model, rng_key).keys()),
    )
    posterior_samples = predictive(random.PRNGKey(1))
    return None, posterior_samples

def sample_parallel_tempering(
    model,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key: random.PRNGKey
):
    def make_nuts_kernel(model, *args, **kwargs):
        return tfp.mcmc.NoUTurnSampler(model, step_size=0.1)

    # Sample the model parameters and latent variables using MCMC
    # kernel = TFPKernel[tfp.mcmc.ReplicaExchangeMC](
    kernel = ReplicaExchangeMC(
        model=model,
        inverse_temperatures=0.9 ** jnp.arange(4),
        make_kernel_fn=make_nuts_kernel,
    )
    mcmc = MCMC(
        sampler=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=1,
    )
    return mcmc, mcmc.run(rng_key)


def get_nested_sampler(
    model,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key: random.PRNGKey
):
    # Sample the model parameters and latent variables using nested sampling
    sampler = NestedSampler(
        model=model,
        constructor_kwargs=dict(
            num_live_points=300,
            max_samples=10_000,
        ),
    )
    return sampler, sampler.run(rng_key, num_samples)

def sample_flow_hmc(
    model,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    rng_key: random.PRNGKey
):
    # Step 1. Train a guide
    guide = autoguide.AutoBNAFNormal(model, num_flows=1, hidden_factors=[3, 3])
    # guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(1e-3), loss=Trace_ELBO())
    # svi_result = svi.run(random.PRNGKey(0), 500)
    svi_state = svi.init(rng_key)
    last_state, losses = lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(100))

    print("SVI loss: ", losses[-1])

    # Step 2. Use trained guide in NeuTra MCMC
    neutra = NeuTraReparam(guide, svi.get_params(last_state))
    # neutra = NeuTraReparam(guide, svi_result.params)
    model_latent = neutra.reparam(model)
    mcmc = MCMC(
        NUTS(model_latent),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )

    # Generate samples and translate back into original space
    mcmc.run(rng_key)
    samples_latent = mcmc.get_samples(group_by_chain=False)["auto_shared_latent"]
    samples = neutra.transform_sample(samples_latent)
    return mcmc, samples


""" Utility functions for the inference. """


def normalize(x):
    return x / jnp.sum(x, axis=-1, keepdims=True)


def fnn_two_layers(X, D_H, D_Y=1, nonlin=jnp.tanh, param_name_prefix="fnn2"):
    """
    A simple two-layer neural network with computational flow
    given by D_X => D_H => D_H => D_Y where D_H is the number of
    hidden units.
    """

    N, D_X = X.shape

    print('X', X[:3, :3])

    # sample first layer (we put unit normal priors on all weights)
    b1 = numpyro.param(f"{param_name_prefix}_b1", jnp.zeros(D_H))
    w1 = numpyro.param(f"{param_name_prefix}_w1", 0.1 * jax.random.normal(jax.random.PRNGKey(1), shape=(D_X, D_H)))
    z1 = nonlin(b1 + jnp.matmul(X, w1))  # <= first layer of activations

    print('z1', z1[:3, :3])

    # sample second layer
    b2 = numpyro.param(f"{param_name_prefix}_b2", jnp.zeros(D_H))
    w2 = numpyro.param(f"{param_name_prefix}_w2", 0.01 * jax.random.normal(jax.random.PRNGKey(1), shape=(D_H, D_H)))
    z2 = nonlin(z1 + b2 + jnp.matmul(z1, w2))  # <= second layer of activations

    print('z2', z2[:3, :3])

    # sample final layer of weights and neural network output
    b3 = numpyro.param(f"{param_name_prefix}_b3", jnp.zeros(D_Y))
    w3 = numpyro.param(f"{param_name_prefix}_w3", 0.01 * jax.random.normal(jax.random.PRNGKey(2), shape=(D_H, D_Y)))
    z3 = b3 + jnp.matmul(z2, w3)  # <= output of the neural network

    print('z3', z3[:3, :3])

    return z3

class SoftmaxTransform(transforms.ParameterFreeTransform):
    """
    Custom softmax transformation for mapping logits to the simplex.
    """

    domain = constraints.real_vector
    codomain = constraints.simplex

    def __call__(self, x):
        """Applies the softmax transformation."""
        return softmax(x, axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """Computes the log-determinant of the Jacobian of the softmax transformation."""
        return -jnp.sum(y * jnp.log(y + 1e-16), axis=-1)  # Avoid log(0) with a small epsilon

    def inverse(self, y):
        """Inverse transform: maps the simplex back to unconstrained space."""
        # return jnp.log(y[..., :-1] + 1e-16) - jnp.log(y[..., -1:] + 1e-16)
        return jnp.log(y)
        # return jnp.log(y + 1e-16)


class Transpose(transforms.ParameterFreeTransform):
    """
    A custom transform to transpose the last two dimensions of a tensor.
    """
    domain = constraints.real_matrix

    def __call__(self, x):
        return x.T

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return 0.0

    def inverse(self, y):
        return y.T


def logistic_normal(name: str, loc, scale):
    return numpyro.sample(name, dist.TransformedDistribution(
        dist.Normal(loc, scale),
        transforms=[StickBreakingTransform()],
    ))

def logistic_multivariate_normal(name: str, loc, covariance_matrix):
    return numpyro.sample(name, dist.TransformedDistribution(
        dist.MultivariateNormal(loc, covariance_matrix),
        transforms=[StickBreakingTransform()],
    ))

