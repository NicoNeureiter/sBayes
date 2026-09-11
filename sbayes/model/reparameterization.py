import jax.numpy as jnp
from numpyro.distributions.transforms import StickBreakingTransform
import numpyro.distributions as dist
import numpyro

LATENT_BOUND = 6.0
"""Half-width of the uniform latent space. Stick-breaking saturates far below this, so
the bound is effectively infinite, but it must be wide enough not to truncate the
Dirichlet prior."""

def dirichlet_from_latent(
    name: str,
    concentration: jnp.ndarray,
    offset: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sample from a Dirichlet distribution through an unconstrained latent space.

    Instead of sampling the simplex directly, a uniform latent vector is sampled and
    mapped to the simplex by a stick-breaking transform. The actual Dirichlet density
    is then added as a factor, corrected for the transform and for the latent's own
    density. This gives the sampler an unconstrained space to move in, which helps for
    low-concentration Dirichlet distributions.

    Args:
        name: the name of the numpyro sample site
        concentration: the concentration parameter of the Dirichlet distribution
        offset: optional point on the simplex to center the latent space on

    Returns:
        The sampled value on the probability simplex.
    """
    n_states_latent = concentration.shape[-1] - 1

    # Sample in latent space, wide enough to cover the tails of the simplex
    x_latent_distr = dist.Uniform(-LATENT_BOUND, LATENT_BOUND).expand((n_states_latent,)).to_event()
    z = jnp.asarray(numpyro.sample(f"{name}_raw", x_latent_distr))

    # The latent density is uniform, so this is a constant, but it is needed to turn
    # the latent density into the Dirichlet density below
    prior_correction_factor = -x_latent_distr.log_prob(z)

    stick_breaking = StickBreakingTransform()

    # Center the latent space on `offset` by shifting it to the latent representation
    x_latent = z if offset is None else z + stick_breaking.inv(offset)

    # Transform to the probability simplex and correct for the transform
    x = jnp.asarray(stick_breaking(x_latent))
    prior_correction_factor += stick_breaking.log_abs_det_jacobian(x_latent, x)

    # Add the Dirichlet density of the transformed value as a factor
    prior_log_prob = dist.Dirichlet(concentration).log_prob(x)
    numpyro.factor(f"{name}_log_prob", prior_log_prob + prior_correction_factor)

    numpyro.deterministic(name, x)

    return x