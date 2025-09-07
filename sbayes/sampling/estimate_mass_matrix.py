import numpy as np
from jax import numpy as jnp

from sbayes.model import Model


def get_mass_matrix_block_structure(model: Model):
    dense_mass = []
    for i in range(model.n_clusters):
        params_clust_i = (f"z_raw_{i}",)
        for p in model.partitions:
            params_clust_i += (f"cluster_effect_raw_{p.name}_{i}",)
        dense_mass.append(params_clust_i)

    # Potentially add weights to the dense mass matrix
    # dense_mass.append(("w",))

    return dense_mass


def create_inv_mass_matrix(model: Model) -> dict:
    n_objects = model.shapes.n_objects
    categorical_partitions = model.data.features.categorical_partitions()

    mm_dim = model.shapes.n_objects
    mm_dim += sum(p.n_features * p.n_states
                  for p in categorical_partitions)

    imm = np.eye(mm_dim)
    i = n_objects
    for p in categorical_partitions:
        i_next = i + p.n_features * p.n_states
        for s in range(p.n_states):
            imm[:n_objects, i+s:i_next:p.n_states][p.values == s] = 0.02
            imm[:n_objects, i+s:i_next:p.n_states][(p.values != s) & ~p.na_values] = -0.02
        imm[:, :n_objects] = imm[:n_objects, :].T
        i = i_next

    imm_by_cluster = {}
    for i in range(model.n_clusters):
        params_clust_i = (f"z_raw_{i}",)
        for p in categorical_partitions:
            params_clust_i += (f"cluster_effect_raw_{p.name}_{i}",)

        imm_by_cluster[params_clust_i] = jnp.array(imm)

    return imm_by_cluster


def split_mass_matrix(imm: dict, param_shapes: dict):
    assert len(imm) == 1.0
    params_tuple = next(iter(imm.keys()))
    imm_diag = imm[params_tuple]

    imm_dict = {}
    i = 0
    for p in params_tuple:
        shape = param_shapes[p]
        k = np.prod(shape[1:])
        imm_dict[p] = imm_diag[:, i:i+k]
        i += k

    assert i == imm_diag.shape[1]

    return imm_dict


def fix_inv_mass_matrix_diag(imm_diag, model, param_shapes) -> dict:
    imm_by_cluster = create_inv_mass_matrix(model)

    all_cluster_params = sum(imm_by_cluster.keys(), ())
    imm_diag = split_mass_matrix(imm_diag, param_shapes)

    imm = {}
    for params_z, imm_z in imm_by_cluster.items():
        imm_diag_z = jnp.concatenate(tuple(imm_diag[p] for p in params_z), axis=1)
        imm_diag_z = jnp.mean(imm_diag_z, axis=0)
        jnp.fill_diagonal(imm_z, imm_diag_z, inplace=False)
        imm[params_z] = imm_z

    for p, imm_diag_p in imm_diag.items():
        if p not in all_cluster_params:
            imm[(p,)] = jnp.mean(imm_diag_p, axis=0)
    return imm
