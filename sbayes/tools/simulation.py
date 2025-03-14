from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase as ABC
from typing import Iterable

import pandas as pd
import numpy as np
import os
import random

import tables
import yaml
import json
import string
from collections import OrderedDict, namedtuple, defaultdict
import argparse

from numpy._typing import NDArray

from sbayes.config.config import SBayesConfig, ModelConfig, PriorConfig
from sbayes.load_data import Objects, Confounder, Features, FeatureType, Data
from sbayes.model import Model, WeightsPrior, ClusterEffectPrior, ClusterPrior
from sbayes.model.prior import CategoricalConfoundingEffectsPrior
from sbayes.preprocessing import sample_categorical, ComputeNetwork
from scipy.stats import invgamma, gamma, kstest, binom

from sbayes.sampling.loggers import SourceLogger
from sbayes.sampling.state import Sample
from sbayes.util import set_experiment_name, fix_relative_path, format_cluster_columns, normalize_weights


def set_simulation_name(feat_meta: dict):
    """ Define the name of the simulation
            Args:
                feat_meta (dict): meta information about the simulated features
            Returns:
                (str): the name of the simulation"""
    sim_name = set_experiment_name()
    for ft in ["categorical", "gaussian", "poisson"]:
        if ft in feat_meta.keys():
            sim_name = "_".join([sim_name, ft])
    return sim_name


def simulate_objects(n: int):
    """ Simulate objects with attributes location (x,y) and ID
        Args:
            n (int): the number of simulated objects
        Returns:
            (Objects): the simulated objects"""

    # Simulate n objects with random locations
    df = pd.DataFrame().assign(
        x=np.random.rand(n),
        y=np.random.rand(n),
        id=range(n),
        name=["".join(['object', str(i)]) for i in range(n)])

    return Objects.from_dataframe(df)


def simulate_confounder_assignment(ob: Objects, conf_meta: dict) -> dict[str, Confounder]:
    """ Randomly assign objects to confounders
        Args:
            ob: simulated objects with attributes location (x,y) and ID
            conf_meta: confounder names, their applicable groups and relative frequency
        Returns:
            The simulated assignment of objects to confounders"""

    df = pd.DataFrame()

    for cf, g in conf_meta.items():
        groups = list(g.keys())
        p = list(g.values())
        df[cf] = np.random.choice(groups, size=ob.n_objects, p=p)

    conf = OrderedDict()
    for c in df.columns.tolist():
        conf[c] = Confounder.from_dataframe(data=df, confounder_name=c)

    return conf


def simulate_uniform_clusters(
    n_clusters: int,
    n_objects: int,
    min_size: int = 1,
    max_size: int = None,
) -> NDArray[bool]:  # (n_clusters, n_objects)
    """Randomly assign each language to one of the areas or no area (equal probability for every option)."""

    # if (1+n_clusters)**n_objects > 1E9:
    #     : Caching

    eye = np.eye(n_clusters+1, dtype=bool)[:, :-1]
    clusters = np.zeros((n_clusters, n_objects), dtype=bool)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    while np.any(cluster_sizes < min_size) or np.any(cluster_sizes > max_size):
        areas_int = np.random.randint(n_clusters + 1, size=n_objects)
        clusters = eye[areas_int].T
        cluster_sizes = np.sum(clusters, axis=1)
    return clusters


def simulate_cluster_assignment(
    objects: Objects,
    n_clusters: int,
    prior: ClusterPrior,
) -> NDArray[bool]:  # (n_clusters, n_objects)
    """ Randomly assign objects to mutually exclusive clusters
        Args:
            objects: simulated objects with attributes location (x,y) and ID
            n_clusters: clusters and their relative frequencies
        Returns:
            The simulated assignment of objects to clusters"""
    if prior.prior_type == prior.PriorType.UNIFORM_AREA:
        return simulate_uniform_clusters(n_clusters, len(objects), prior.min, prior.max)
    else:
        raise NotImplementedError(f'Simulation for prior type {prior.prior_type} not implemented yet')


def define_names(feat: dict):
    """ Set the feature names and state names for categorical features
        Args:
            feat (dict): number of features per feature type
        Returns:
            (dict): The feature and state name
    """
    # Add the features and remember their names
    fi = 1
    feat_names = {}
    for ft, nf in feat.items():

        if ft == 'categorical':
            feat_names[ft] = {}
            for f in nf:
                for nf_cat in range(f[1]):
                    feat_names[ft]["".join(['f', str(fi)])] = [string.ascii_uppercase[i] for i in range(f[0])]
                    fi += 1

        if ft == 'gaussian':
            feat_names[ft] = ["".join(['f', str(fi + i)]) for i in range(nf)]
            fi += nf

        if ft == 'poisson':
            feat_names[ft] = ["".join(['f', str(fi + i)]) for i in range(nf)]
            fi += nf

    return feat_names


def simulate_weights_prior(feat: dict, conf: dict, p_hyper: dict):
    """ Simulate a Dirichlet weights prior, explicitly writing it out for each feature
        Args:
            feat (dict): number of features per feature type
            conf (dict): the confounders of the model
            p_hyper (dict): probabilities of the hyperparameter values for simulating the weights prior
        Returns:
            (dict): a Dirichlet weights prior for each feature, structured per feature type"""

    n_components = len(conf) + 1
    prior = {}
    for ft, fs in feat.items():
        prior[ft] = {}
        if ft == 'categorical':
            n_f = sum([i[1] for i in fs])
        else:
            n_f = fs

        concentration = np.full(shape=(n_f, n_components), fill_value=1.)
        assignment = np.array(random.choices(range(n_components), k=n_f))
        concentration[np.arange(n_f), assignment] = np.random.uniform(p_hyper['concentration'][0],
                                                                      p_hyper['concentration'][1], size=n_f)

        prior[ft]['concentration'] = concentration

    return prior


def simulate_confounding_effect_prior(feat: dict, conf: dict, p_hyper: dict):
    """ Simulate the prior distributions per confounding effect, explicitly writing it out for each feature
        Args:
            feat (dict): Number of features per feature type
            conf (dict): the confounders of the model
            p_hyper (dict): probabilities of the hyperparameter values for simulating the confounding effect prior
        Returns:
            (dict): confounding effect prior for each feature, structured per feature type"""

    prior = {}

    for ft, fs in feat.items():
        prior[ft] = {}
        for c, groups in conf.items():
            prior[ft][c] = {}
            for g in groups.keys():
                if ft == 'categorical':

                    n_states_per_categorical_feat = [i for i, c in feat['categorical'] for _ in range(c)]

                    concentration = []
                    for n_states_f in n_states_per_categorical_feat:
                        concentration_f = np.full(shape=n_states_f, fill_value=1.)
                        assignment_f = np.array(random.choices(range(n_states_f), k=1))
                        concentration_f[assignment_f] = np.random.uniform(p_hyper[ft]['concentration'][0],
                                                                          p_hyper[ft]['concentration'][1],
                                                                          size=1)
                        concentration.append(concentration_f)
                    prior[ft][c][g] = dict(concentration=concentration)

                elif ft == 'gaussian':
                    mean = dict(mu_0=np.random.uniform(p_hyper[ft]['mean']['mu_0'][0],
                                                       p_hyper[ft]['mean']['mu_0'][1], size=fs),
                                sigma_0=np.random.uniform(p_hyper[ft]['mean']['sigma_0'][0],
                                                          p_hyper[ft]['mean']['sigma_0'][1], size=fs))
                    variance = dict(alpha_0=np.random.uniform(p_hyper[ft]['variance']['alpha_0'][0],
                                                              p_hyper[ft]['variance']['alpha_0'][1], size=fs),
                                    beta_0=np.random.uniform(p_hyper[ft]['variance']['beta_0'][0],
                                                             p_hyper[ft]['variance']['beta_0'][1], size=fs))
                    prior[ft][c][g] = dict(mean=mean,
                                           variance=variance)

                elif ft == 'poisson':
                    rate = dict(alpha_0=np.random.uniform(p_hyper[ft]['rate']['alpha_0'][0],
                                                          p_hyper[ft]['rate']['alpha_0'][1], size=fs),
                                beta_0=np.random.uniform(p_hyper[ft]['rate']['beta_0'][0],
                                                         p_hyper[ft]['rate']['beta_0'][1], size=fs))
                    prior[ft][c][g] = dict(rate=rate)

                else:
                    raise ValueError("Feature type nor supported")

    return prior


def simulate_cluster_effect_prior(feat: dict,  n_clusters: int, p_hyper: dict):
    """ Simulate the prior distributions for the cluster effect, explicitly writing it out for each feature
        Args:
            feat (dict): number of features per feature type
            n_clusters (int): the number of clusters
            p_hyper (dict): probabilities of the hyperparameter values for simulating the cluster effect prior
        Returns:
            (dict): cluster effect prior for each feature, structured per feature type """

    prior = {}
    for ft, fs in feat.items():
        prior[ft] = {}
        for i_clust in range(n_clusters):
            cl = f'cluster_{i_clust}'
            if ft == 'categorical':
                n_states_per_categorical_feat = [i for i, c in feat['categorical'] for _ in range(c)]
                concentration = []
                for n_states_f in n_states_per_categorical_feat:
                    concentration_f = np.full(shape=n_states_f, fill_value=1.)
                    assignment_f = np.array(random.choices(range(n_states_f), k=1))
                    concentration_f[assignment_f] = np.random.uniform(p_hyper[ft]['concentration'][0],
                                                                      p_hyper[ft]['concentration'][1],
                                                                      size=1)
                    concentration.append(concentration_f)
                    prior[ft][cl] = dict(concentration=concentration)
            elif ft == 'gaussian':
                mean = dict(mu_0=np.random.uniform(p_hyper[ft]['mean']['mu_0'][0],
                                                   p_hyper[ft]['mean']['mu_0'][1], size=fs),
                            sigma_0=np.random.uniform(p_hyper[ft]['mean']['sigma_0'][0],
                                                      p_hyper[ft]['mean']['sigma_0'][1], size=fs))
                variance = dict(alpha_0=np.random.uniform(p_hyper[ft]['variance']['alpha_0'][0],
                                                          p_hyper[ft]['variance']['alpha_0'][1], size=fs),
                                beta_0=np.random.uniform(p_hyper[ft]['variance']['beta_0'][0],
                                                         p_hyper[ft]['variance']['beta_0'][1], size=fs))
                prior[ft][cl] = dict(mean=mean, variance=variance)
            elif ft == 'poisson':
                rate = dict(alpha_0=np.random.uniform(p_hyper[ft]['rate']['alpha_0'][0],
                                                      p_hyper[ft]['rate']['alpha_0'][1], size=fs),
                            beta_0=np.random.uniform(p_hyper[ft]['rate']['beta_0'][0],
                                                     p_hyper[ft]['rate']['beta_0'][1], size=fs))
                prior[ft][cl] = dict(rate=rate)
            else:
                raise ValueError("Feature type nor supported")
    return prior


def simulate_weights(prior: WeightsPrior, fix_weights: bool)-> dict[FeatureType, NDArray[float]]:
    """ Simulates weights of the cluster and confounding effects for all features
    Args:
        prior: The Dirichlet weights prior for each feature, structured per feature type
        fix_weights: Whether to use fixed weights of 1/n_components for each feature.

    Returns:
        Simulated weights for each effect and each feature, structured per feature type"""

    weights = {}

    for ft, ft_weights_prior in prior.ft_priors.items():
        w = []
        for f in ft_weights_prior.concentration:
            w.append(np.random.dirichlet(f))
        weights[ft] = np.vstack(w)

        if fix_weights:
            weights[ft] = np.ones_like(weights[ft]) / weights[ft].shape[-1]

    return weights


def get_has_components(clusters: NDArray[bool], confounders: Iterable[Confounder]) -> NDArray[bool]:
    """Find which objects are assigned to one of the clusters/groups in each component."""
    has_component = [np.any(clusters, axis=0)]
    for conf in confounders:
        has_component.append(np.any(conf.group_assignment, axis=0))
    return np.array(has_component).T


def simulate_source(
    weights_sim: dict[FeatureType, NDArray[float]],
    clust_sim: NDArray[bool],                       # (n_clusters, n_objects)
    conf_sim: dict[str, Confounder],
) -> dict[FeatureType, NDArray[bool]]:
    """Simulate the assignment of each observation (object and feature) to a component (cluster or confounder).

    Args:
        weights_sim: Simulated weights for each effect and each feature, structured per feature type
        clust_sim: The simulated assignment of objects to clusters
        conf_sim: The simulated assignment of objects to confounder groups
    Returns:
        The simulated source assignment matrix for each feature type.
    """
    source = {}
    for ft, weights in weights_sim.items():
        has_components = get_has_components(clust_sim, conf_sim.values())
        weights_normalized = normalize_weights(weights, has_components)
        source[ft] = sample_categorical(weights_normalized, binary_encoding=True)
    return source


def simulate_confounding_effect(model: Model):
    """ Simulates the confounding effect per confounder for all features
        Args:
            prior (dict): The confounding effect prior for each feature, structured per feature type

        Returns:
            (dict): simulated confounding effect per confounder for each feature, structured per feature type"""
    prior = model.prior.confounding_effects_prior


    conf_effect = defaultdict(dict)
    for c, prior_c in prior.items():
        for ft, prior_c_ft in prior_c.ft_priors.items():
            conf_effect[ft][c] = {}
            for i_g, g in enumerate(model.data.confounders[c].group_names):
                if ft == 'categorical':
                    concentration = prior_c_ft.concentration[g]
                    max_categorical_states = model.shapes.n_states_categorical
                    p = []
                    for f in concentration:
                        p_f = np.zeros(max_categorical_states)
                        p_f[:len(f)] = np.random.dirichlet(f, size=1)
                        p.append(p_f)
                    conf_effect[ft][c][g] = np.vstack(p)
                elif ft == 'gaussian':
                    mean_prior = prior_c_ft.mean
                    mean = np.random.normal(mean_prior.mu_0_array[i_g], mean_prior.sigma_0_array[i_g])

                    var_prior = prior_c_ft.variance
                    if var_prior.group_prior_type[g] == var_prior.PriorType.FIXED:
                        variance = np.full(prior[c].shapes.n_features_gaussian, fill_value=var_prior.fixed_value[g])
                    else:
                        variance = invgamma.rvs(var_prior.config[g].parameters['alpha_0'],
                                                var_prior.config[g].parameters['beta_0'])

                    conf_effect[ft][c][g] = {'mean': mean, 'variance': variance}
                elif ft == 'poisson':
                    rate = gamma.rvs(prior_c_ft.alpha_0_array[i_g], prior_c_ft.beta_0_array[i_g])
                    conf_effect[ft][c][g] = dict(rate=rate)
                else:
                    raise ValueError("Feature type not supported")

    #
    # for ft, conf in prior.items():
    #     conf_effect[ft] = {}
    #     for c, groups in conf.items():
    #         conf_effect[ft][c] = {}
    #         for g, params in groups.items():
    #             conf_effect[ft][c][g] = {}
    #             if ft == 'categorical':
    #                 max_categorical_states = \
    #                     max(len(i) for i in next(iter(next(iter(prior['categorical'].values())).
    #                                                   values()))['concentration'])
    #                 p = []
    #                 for f in params['concentration']:
    #                     p_f = np.zeros(max_categorical_states)
    #                     p_f[:len(f)] = np.random.dirichlet(f, size=1)
    #                     p.append(p_f)
    #                 conf_effect[ft][c][g] = np.vstack(p)
    #             elif ft == 'gaussian':
    #                 mean = np.random.normal(params['mean']['mu_0'], params['mean']['sigma_0'])
    #                 # variance = invgamma.rvs(params['variance']['alpha_0'], params['variance']['beta_0'])
    #                 variance = invgamma.mean(params['variance']['alpha_0'], params['variance']['beta_0'])
    #                 conf_effect[ft][c][g] = dict(mean=mean,
    #                                              variance=variance)
    #             elif ft == 'poisson':
    #                 rate = gamma.rvs(params['rate']['alpha_0'], params['rate']['beta_0'])
    #                 conf_effect[ft][c][g] = dict(rate=rate)
    #             else:
    #                 raise ValueError("Feature type not supported")

    return conf_effect


def simulate_cluster_effect(prior: ClusterEffectPrior):
    """ Simulates the cluster effect for all features
        Args:
            prior (dict): The cluster effect prior for each feature, structured per feature type

        Returns:
            (dict): simulated cluster effect for each feature, structured per feature type"""

    clust_effect = {}
    for ft in FeatureType.values():
        clust_effect[ft] = {}
        for i_clust in range(prior.shapes.n_clusters):
            cl = f'cluster_{i_clust}'
            if ft == FeatureType.categorical:
                if prior.categorical is None:
                    continue
                # max_categorical_states = max(len(i) for i in next(iter(prior.categorical.values()))['concentration'])
                max_categorical_states = prior.categorical.shapes.n_states_categorical
                p = []
                for f in prior.categorical.concentration:
                    p_f = np.zeros(max_categorical_states)
                    p_f[: len(f)] = np.random.dirichlet(f, size=1)
                    p.append(p_f)
                clust_effect[ft][cl] = np.vstack(p)

            elif ft == FeatureType.gaussian:
                if prior.gaussian is None:
                    continue
                mean_prior = prior.gaussian.mean
                var_prior = prior.gaussian.variance
                mean = np.random.normal(mean_prior.mu_0_array, mean_prior.sigma_0_array)
                if var_prior.prior_type is var_prior.PriorType.FIXED:
                    variance = np.full(prior.shapes.n_features_gaussian, fill_value=var_prior.fixed_value)
                else:
                    variance = invgamma.rvs(var_prior.config.parameters['alpha_0'], var_prior.config.parameters['beta_0'])

                clust_effect[ft][cl] = dict(mean=mean,
                                            variance=variance)

            elif ft == FeatureType.poisson:
                if prior.poisson is None:
                    continue
                rate = gamma.rvs(prior.poisson.alpha_0_array, prior.poisson.beta_0_array)
                clust_effect[ft][cl] = dict(rate=rate)
            elif ft == FeatureType.logitnormal:
                pass
            else:
                raise ValueError(f"Feature type `{ft}` not supported")

    return clust_effect


def simulate_features(
    weights: dict[FeatureType, NDArray[float]],
    conf_effect: dict[FeatureType, dict[str, NDArray[float]]],
    clust_effect: dict[FeatureType, NDArray[float]],
    conf: dict[str, Confounder],
    clust: NDArray[bool],
    source: dict[FeatureType, NDArray[bool]],
    names: dict[FeatureType, dict],
):
    """Simulate features.
        Args:
            weights: the influence of the confounding and cluster effects for each feature
            conf_effect: the confounding effect per confounder for each feature
            clust_effect: the cluster effect for each feature
            conf: the assignments of sites to groups of a confounder
            clust (np.ndarray): the assignment of sites to clusters
                shape: (n_clusters, n_objects)
            names: The feature and state names

        Returns:
            (dict): The simulated features per feature type"""

    # Which object is assigned to which effect?
    n_objects = clust.shape[1]
    effect_assignment = [np.any(clust, axis=0)]
    assignment_order = {"cluster": 0}

    for i, (n, c) in enumerate(conf.items()):
        effect_assignment.append(np.any(c.group_assignment, axis=0))
        assignment_order[n] = i + 1

    feature_types = list(names.keys())
    feat = {}

    # Iterate over each feature type
    for ft in feature_types:

        # Are the weights fine?
        assert np.allclose(a=np.sum(weights[ft], axis=-1), b=1.)

        if ft == 'categorical':
            n_feat = len(names[ft].keys())
        else:
            n_feat = len(names[ft])

        # Normalize the weights for each object depending on whether clusters or confounder are relevant for it
        normed_weights = np.transpose(normalize_weights(weights[ft], np.array(effect_assignment).T), (1, 0, 2))

        if ft == 'categorical':

            feat[ft] = np.zeros((n_objects, n_feat), dtype=str)

            for f, f_name in enumerate(names[ft]):

                p_cl = np.array(list(clust_effect[ft].values()))[:, f, :]
                # Compute the feature likelihood matrix (for all objects and all states)
                lh_p_cl = clust.T.dot(p_cl).T
                lh_p = source[ft][:, f, assignment_order['cluster']] * lh_p_cl
                for n, c in conf.items():
                    p_c = np.array(list(conf_effect[ft][n].values()))[:, f, :]
                    lh_p_c = c.group_assignment.T.dot(p_c).T
                    lh_p += source[ft][:, f, assignment_order[n]] * lh_p_c
                # Sample from the categorical distribution defined by lh_feature
                sample = sample_categorical(lh_p.T)
                # Replace integers with actual state names
                feat[ft][:, f] = np.array([names[ft][f_name][x] for x in sample])

        elif ft == 'gaussian':

            feat[ft] = np.zeros((n_objects, n_feat), dtype=float)

            for f, f_name in enumerate(names[ft]):

                mean_cl = np.array([cl['mean'] for cl in clust_effect[ft].values()])[:, f]
                variance_cl = np.array([cl['variance'] for cl in clust_effect[ft].values()])[:, f]

                # Compute the feature likelihood matrix (for all objects)
                lh_mean_cl = clust.T.dot(mean_cl).T
                lh_variance_cl = clust.T.dot(variance_cl).T

                lh_mean = np.zeros_like(lh_mean_cl)
                lh_variance = np.zeros_like(lh_variance_cl)

                s = source[ft][:, f, assignment_order['cluster']]
                lh_mean[s] = lh_mean_cl[s]
                lh_variance[s] = lh_variance_cl[s]

                for n, c in conf.items():
                    mean_c = np.array([g['mean'] for g in conf_effect[ft][n].values()])[:, f]
                    variance_c = np.array([g['variance'] for g in conf_effect[ft][n].values()])[:, f]

                    lh_mean_c = c['group_assignment'].T.dot(mean_c).T
                    lh_variance_c = c['group_assignment'].T.dot(variance_c).T

                    s = source[ft][:, f, assignment_order[n]]
                    assert np.all(lh_mean[s] ==0)
                    lh_mean[s] = lh_mean_c[s]
                    assert np.all(lh_variance[s] == 0)
                    lh_variance[s] = lh_variance_c[s]

                feat[ft][:, f] = np.random.normal(lh_mean, lh_variance)

        elif ft == 'poisson':

            feat[ft] = np.zeros((n_objects, n_feat), dtype=int)

            for f, f_name in enumerate(names[ft]):

                rate_cl = np.array([cl['rate'] for cl in clust_effect[ft].values()])[:, f]
                # Compute the feature likelihood matrix (for all objects)
                lh_rate_cl = clust.T.dot(rate_cl).T

                lh_rate = source[ft][:, f, assignment_order['cluster']] * lh_rate_cl

                for n, c in conf.items():
                    rate_c = np.array([g['rate'] for g in conf_effect[ft][n].values()])[:, f]
                    lh_rate_c = c['group_assignment'].T.dot(rate_c).T

                    lh_rate += source[ft][:, f, assignment_order[n]] * lh_rate_c

                feat[ft][:, f] = np.random.poisson(lh_rate)

    return feat


def combine_data(ob: Objects, conf: OrderedDict, feat: dict):
    """ Combine the simulated objects, confounders and features in a DataFrame
        Args:
            ob (Objects): simulated objects with attributes location (x,y) and ID
            conf (OrderedDict): simulated confounders
            feat (dict): simulated features
        Returns:
            (pd.DataFrame, list): a DataFrame with the simulated objects, confounders and features; the feature names"""

    # Add the objects
    df = pd.DataFrame().assign(
        x=ob.locations[:, 0],
        y=ob.locations[:, 1],
        id=ob.id,
        name=ob.names)

    # Reverse the one-hot encoding of the confounders
    for n, cf in conf.items():
        reversed_encoding = []
        for col in cf['group_assignment'].T:
            index = np.where(col)[0]
            if len(index) == 0:
                reversed_encoding.append(None)
            else:
                reversed_encoding.append(cf['group_names'][index[0]])
        df[n] = reversed_encoding

    # Add the features and remember their names
    fi = 1
    feat_names = {}
    for ft, fs in feat.items():
        feat_names[ft] = []
        for f in fs.T:
            fn = "".join(['f', str(fi)])
            feat_names[ft].append(fn)
            df[fn] = f
            fi += 1
    return df, feat_names


def export_priors(priors=None, sim_name=None, names=None, path=None):
    """ Export the simulated priors
            Args:
                priors (dict): the simulated prior
                sim_name (str): (folder) name of the simulation
                names (dict): simulated names of the features and states
                path (Path): path for storing the simulation
            Returns:
                (str: the simulation name): """
    os.makedirs(path, exist_ok=True)
    write_priors(priors, names, path)
    return sim_name


def export_parameters(params=None, names=None, path=None):
    """ Export the simulated data and parameters
        Args:
            params (dict): the parameters to simulate the data
            names (dict): simulated names of the features and states
            path (Path): folder path for storing the simulation
            """
    os.makedirs(str(path), exist_ok=True)

    # Export the parameters as stats_sim.txt
    write_parameters(params=params,
                     names=names,
                     path=path / 'stats_sim.txt')

    # Export the clusters as cluster_sim.txt
    with open(path / 'clusters_sim.txt', "w") as file:
        file.write(format_cluster_columns(params['clusters']))

    # Export the source array
    source = params['source']
    source_file = tables.open_file(path / f'source_sim.h5', mode="w")
    for ft, source_ft in source.items():
        logged_source_ft = source_file.create_earray(
            where=source_file.root,
            name=f"source_{ft.value}",
            atom=tables.BoolCol(),
            filters=tables.Filters(
                complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
            ),
            shape=(0, ) + source_ft.shape,
        )
        logged_source_ft.append(source_ft[None, ...])
    source_file.close()


def export_meta(meta: dict, path: Path = None):
    """ Export the simulated data and parameters
            Args:
                meta: the simulated the data
                path: folder path for storing the simulation
                """
    folder = fix_relative_path(path, os.getcwd()) or os.getcwd()
    os.makedirs(folder, exist_ok=True)

    with open(folder / 'meta_sim.yaml', 'w') as file:
        yaml.safe_dump(meta, file, sort_keys=False)


def export_data(data: pd.DataFrame, path: Path = None):
    """ Export the simulated data and parameters
        Args:
            data: the simulated the data
            path: folder path for storing the simulation
    """
    folder = fix_relative_path(path, os.getcwd()) or os.getcwd()
    os.makedirs(folder, exist_ok=True)

    # Export the data
    data.to_csv(folder / 'features_sim.csv', index=False)


def write_parameters(params, names, path):
    """ Structure the parameters and write them to a text file
            Args:
                params (dict): the parameters to simulate the data
                names (dict): simulated names of the features and states
                path (Path): path for saving the text file
            Returns:
    """
    feat_names = dict(
        categorical=list(names['categorical'].keys()) if 'categorical' in names.keys() else None,
        gaussian=names['gaussian'] if 'gaussian' in names.keys() else None,
        poisson=names['poisson'] if 'poisson' in names.keys() else None
    )

    column_names = []
    row = {}
    for ft, weights in params['weights'].items():
        for i_f, f in enumerate(feat_names[ft]):

            # Cluster effect weights
            name = f"w_cluster_{f}"
            column_names += [name]
            row[name] = weights[i_f][0]

            # Confounding effect weights
            for i_conf, conf in enumerate(params['confounders'].keys()):
                name = f"w_{conf}_{f}"
                column_names += [name]
                # index of confounding effect starts with 1
                row[name] = weights[i_f][i_conf+1]

            # Cluster effect
            for i_cl, (_, cl) in enumerate(params['cluster_effects'][ft].items()):
                if ft == "categorical":
                    for i_s, s in enumerate(names[ft][f]):
                        name = f"cluster_{i_cl}_{f}_{s}"
                        column_names += [name]
                        row[name] = cl[i_f][i_s]

                elif ft == "gaussian":
                    name = f"cluster_{i_cl}_{f}_mu"
                    column_names += [name]
                    row[name] = cl['mean'][i_f]

                    name = f"cluster_{i_cl}_{f}_sigma"
                    column_names += [name]
                    row[name] = cl['variance'][i_f]

                elif ft == "poisson":
                    name = f"cluster_{i_cl}_{f}_lambda"
                    column_names += [name]
                    row[name] = cl['rate'][i_f]

            # Confounding effects
            for conf_n, conf in params['confounding_effects'][ft].items():
                for i_g, (g_name, g) in enumerate(conf.items()):
                    if ft == "categorical":
                        for i_s, s in enumerate(names[ft][f]):
                            name = f"{conf_n}_{g_name}_{f}_{s}"
                            column_names += [name]
                            row[name] = g[i_f][i_s]

                    elif ft == "gaussian":
                        name = f"{conf_n}_{g_name}_{f}_mu"
                        column_names += [name]
                        row[name] = g['mean'][i_f]
                        name = f"{conf_n}_{g_name}_{f}_sigma"
                        column_names += [name]
                        row[name] = g['variance'][i_f]

                    elif ft == "poisson":
                        name = f"{conf_n}_{g_name}_{f}_lambda"
                        column_names += [name]
                        row[name] = g['rate'][i_f]

    float_format = "%.12g"
    row_str = "\t".join([float_format % row[k] for k in column_names])

    with open(path, "w") as file:
        file.write("\t".join(column_names) + "\n")
        file.write(row_str + "\n")
        file.close()


def write_priors(priors, names, folder):
    """ Structure the parameters and write them to a text file
               Args:
                   priors (dict): the simulated priors
                   names (dict): simulated names of the features and states
                   folder (Path): path for saving the priors
               Returns:
                   """
    float_style = '%.3f'
    ext = 'prior_sim.yaml'
    assignment_order = ["cluster"]
    any_ft = list(priors['confounding_effect_prior'].keys())[0]
    for c in priors['confounding_effect_prior'][any_ft].keys():
        assignment_order.append(c)

    feat_names = dict(
        categorical=list(names['categorical'].keys()) if 'categorical' in names.keys() else None,
        gaussian=names['gaussian'] if 'gaussian' in names.keys() else None,
        poisson=names['poisson'] if 'poisson' in names.keys() else None
    )

    # Weights prior
    os.makedirs(folder / "weights", exist_ok=True)

    for ft, weights in priors['weights_prior'].items():
        prior_yaml = {}

        for i_f, f in enumerate(weights['concentration']):
            f_n = feat_names[ft][i_f]
            prior_yaml[f_n] = {}
            for i_s, s in enumerate(f):
                prior_yaml[f_n][assignment_order[i_s]] = float_style % s

        path = folder / "_".join(['weights/weights', ft, ext])

        with open(path, 'w') as file:
            yaml.safe_dump(prior_yaml, file, sort_keys=False)

    # # Cluster effect prior
    # os.makedirs(folder / "cluster_effect", exist_ok=True)
    # for ft, clust in priors['cluster_effect_prior'].items():
    #     for cl, cl_eff in clust.items():
    #         prior_yaml = {}
    #         for i_f, f in enumerate(feat_names[ft]):
    #
    #             if ft == 'categorical':
    #
    #                 prior_yaml[f] = \
    #                     dict(concentration={n: float_style % c for n, c in zip(names['categorical'][f],
    #                                                                            cl_eff['concentration'][i_f])})
    #             elif ft == 'gaussian':
    #                 prior_yaml[f] = dict(mean=dict(mu_0=float_style % cl_eff['mean']['mu_0'][i_f],
    #                                                sigma_0=float_style % cl_eff['mean']['sigma_0'][i_f]))
    #             elif ft == 'poisson':
    #                 prior_yaml[f] = dict(rate=dict(alpha_0=float_style % cl_eff['rate']['alpha_0'][i_f],
    #                                                beta_0=float_style % cl_eff['rate']['beta_0'][i_f]))
    #
    #         path = folder / 'cluster_effect' / "_".join(['cluster_effect', cl, ft, ext])
    #         with open(path, 'w') as file:
    #             yaml.safe_dump(prior_yaml, file, sort_keys=False)

    # Confounding effect prior
    os.makedirs(folder / "confounding_effects", exist_ok=True)
    for ft, confounder, in priors['confounding_effect_prior'].items():
        for conf, conf_eff in confounder.items():
            for g, g_eff in conf_eff.items():
                prior_yaml = {}
                for i_f, f in enumerate(feat_names[ft]):
                    if ft == 'categorical':
                        prior_yaml[f] = \
                            dict(concentration={n: float_style % c for n, c in zip(names['categorical'][f],
                                                                                   g_eff['concentration'][i_f])})
                    elif ft == 'gaussian':
                        prior_yaml[f] = dict(mean=dict(mu_0=float_style % g_eff['mean']['mu_0'][i_f],
                                                       sigma_0=float_style % g_eff['mean']['sigma_0'][i_f]))
                    elif ft == 'poisson':
                        prior_yaml[f] = dict(rate=dict(alpha_0=float_style % g_eff['rate']['alpha_0'][i_f],
                                                       beta_0=float_style % g_eff['rate']['beta_0'][i_f]))

                path = folder / 'confounding_effects' / '_'.join([conf, g, ft, ext])
                with open(path, 'w') as file:
                    yaml.safe_dump(prior_yaml, file, sort_keys=False)


def simulate_prior(meta, hyper, path):
    """Simulate the priors and write to file.
            Args:
                meta (dict): meta information about the objects, clusters, features and confounding effects
                hyper(dict): hyperparameters for defining the priors
                path(Path): file path to store the results

            Returns:
                (dict): the simulated prior"""

    # Simulate hyperparameters to define the weights prior
    weights_prior_sim = simulate_weights_prior(meta['features'], meta['confounders'], hyper['weights'])

    # Simulate hyperparameters to define the confounding effect prior
    conf_prior_sim = simulate_confounding_effect_prior(meta['features'], meta['confounders'],
                                                       hyper['confounding_effects'])

    export_priors(priors=dict(weights_prior=weights_prior_sim,
                              confounding_effect_prior=conf_prior_sim),
                  names=meta['names'], path=path)

    return dict(weights=weights_prior_sim, confounding_effects=conf_prior_sim)


def simulate_parameters(
    model: Model,
    meta: dict = None,
    path: Path = None,
    fix_weights: bool = True
) -> dict[str, object]:
    """Simulate the parameters and write to file
        Args:
            model: the sBayes model containing information on the prior distributions
            prior_sim: simulated prior
            meta: meta information about the objects, clusters, features and confounding effects
            path: file path to store the results
        Returns:
            the simulated parameters
    """
    data = model.data

    # Set the assignment to clusters
    clust_sim = simulate_cluster_assignment(data.objects, model.shapes.n_clusters, model.prior.cluster_prior)

    # Simulate the weights
    weights_sim = simulate_weights(model.prior.weights_prior, fix_weights)

    # Simulate the source assignment
    source_sim = simulate_source(weights_sim, clust_sim, data.confounders)

    # Simulate the confounding effect
    conf_effects_sim = simulate_confounding_effect(model)

    # Simulate the cluster effect
    clust_effect_sim = simulate_cluster_effect(model.prior.cluster_effect_prior)

    if path:
        export_parameters(params=dict(weights=weights_sim,
                                      clusters=clust_sim,
                                      confounders=data.confounders,
                                      features=meta['features'],
                                      cluster_effects=clust_effect_sim,
                                      confounding_effects=conf_effects_sim,
                                      source=source_sim,
                                      ),
                          names=meta['names'],
                          path=path)

    return dict(objects=data.objects, confounders=data.confounders, clusters=clust_sim,
                weights=weights_sim, confounding_effects=conf_effects_sim,
                cluster_effect=clust_effect_sim, source=source_sim)


def simulate_data(
    param_sim: dict,
    meta: dict,
    path: Path
):
    """Simulate the parameters and write to file.
        Args:
            param_sim: simulated parameters
            meta: meta information about the objects, clusters, features and confounding effects
            path: file path to store the results
    """

    # Simulate the features
    features_sim = simulate_features(param_sim['weights'],
                                     param_sim['confounding_effects'],
                                     param_sim['cluster_effect'],
                                     param_sim['confounders'],
                                     param_sim['clusters'],
                                     param_sim['source'],
                                     meta['names'])

    # Combine the simulated objects, confounders and features in a DataFrame
    data_sim, feature_names_sim = combine_data(param_sim['objects'],
                                               param_sim['confounders'],
                                               features_sim)

    # Export the simulated parameters and data
    export_data(data=data_sim, path=path)

    return data_sim


def write_feature_types(feature_names: dict, path: Path):
    feature_types = {}
    for ft, features_ft in feature_names.items():
        if ft == 'categorical':
            for name_f, states in features_ft.items():
                feature_types[name_f] = {'type': ft, 'states': states}
        else:
            # cfg = model_config.prior.cluster_effect.gaussian.mean.parameters
            for name_f in features_ft:
                feature_types[name_f] = {'type': ft, 'states': {'min': -10_000.0, 'max': 10_000.0}}

    print('Write features types:', feature_types)
    print('To path', path)
    with open(path, 'w') as f:
        yaml.safe_dump(feature_types, f, sort_keys=False)


class DummyGenericTypeFeatures:

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.names = np.arange(self.n_features).astype(str)
        self.na_number = 0
        self.na_values = None


def dummy_confounders(confounders_meta: dict, n_objects: int) -> dict[str, Confounder]:
    return {
        conf: Confounder(
            name=conf,
            group_assignment=np.zeros((len(meta), n_objects), dtype=bool),
            group_names=list(meta.keys())
        )
        for conf, meta in confounders_meta.items()
    }


def dummy_features(features_meta, n_objects):
    ft_args = {}
    for ft in FeatureType.values():
        if ft not in features_meta:
            ft_args[ft] = None
            continue

        ft_meta = features_meta[ft]
        if ft == FeatureType.categorical:
            n_features = sum(x[1] for x in ft_meta)
            n_states = max(x[0] for x in ft_meta)

            states = np.zeros((n_features, n_states), dtype=bool)
            state_names = []
            i_f = 0
            for n_s, n_f in ft_meta:
                for _ in range(n_f):
                    states[i_f, :n_s] = True
                    state_names.append(np.arange(n_s).astype(str))
                    i_f += 1

            ft_args[ft] = CategoricalFeatures(
                values=np.zeros((n_objects, n_features, n_states), dtype=bool),
                states=states,
                state_names=np.arange(n_features).astype(str),
                feature_names=np.arange(n_features).astype(str),
                na_number=0,
            )
        else:
            ft_args[ft] = DummyGenericTypeFeatures(n_features=ft_meta)

    return Features(**ft_args)


class DummyData:
    def __init__(self, n_objects, confounders, features_meta):
        self.objects = Objects.from_dataframe(
            pd.DataFrame({
                'id': np.arange(n_objects).astype(str),
                'x': np.random.random(n_objects),
                'y': np.random.random(n_objects),
            })
        )
        self.features = dummy_features(features_meta, n_objects)
        self.confounders = confounders
        self.crs = None
        self.geo_cost_matrix = None
        self.network = ComputeNetwork(self.objects)
        self.logger = None


def main(name: str, config: Path, n_sim: int):
    if config.suffix.lower() in (".yaml", ".yml"):
        with open(config, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = json.load(config)

    n_objects = 13
    objects_meta = {'n': n_objects}

    # Names of simulated confounders and simulated groups per confounder and their relative frequency (must sum to 1).
    confounders_meta = {
        # conf1=dict(a=0.3, b=0.4, c=0.3),
        # conf2=dict(d=0.5, e=0.5)
        'conf2': {'d': 1.0}
    }

    # Number of simulated features per feature type
    # For categorical features the tuple (2, 3) simulates 3 features with 2 states
    if name == 'test_categorical':
        features_meta = {'categorical': [(3, 5), (4, 20)]}
    elif name == 'test_gaussian':
        features_meta = {'gaussian': 3}
    elif name == 'test_mixed':
        features_meta = {
            'categorical': [(2, 5), (3, 5)],
            'gaussian': 3,
            'poisson': 10,
        }
    else:
        raise ValueError('Define features_meta')

    # Create data and model objects
    model_config = ModelConfig(**config_dict['model'])
    objects = simulate_objects(n_objects)
    confounders = simulate_confounder_assignment(objects, confounders_meta)
    data = Data(objects, dummy_features(features_meta, n_objects), confounders)
    model = Model(data, model_config)

    meta_parameters = dict(objects=objects_meta,
                           confounders=confounders_meta,
                           n_clusters=model_config.clusters,
                           features=features_meta,
                           names=define_names(features_meta))

    base_dir = Path("experiments/simulation")
    simulation_name = name or set_simulation_name(features_meta)

    # Write meta data
    path_meta = base_dir / simulation_name / "meta"
    export_meta(meta=meta_parameters, path=path_meta)
    write_feature_types(meta_parameters['names'], path_meta / 'feature_types.yaml')

    for i in range(n_sim):
        # Simulate the parameters
        path_parameters = base_dir / simulation_name / "parameters" / f"sim_{i+1}"
        parameters_sim = simulate_parameters(model, meta_parameters, path_parameters, fix_weights=True)

        # Simulate the data
        path_data = base_dir / simulation_name / "data" / f"sim_{i+1}"

        print(path_parameters, path_data)
        simulate_data(parameters_sim, meta_parameters, path_data)

        # samples = []
        # while len(samples) < 200:
        #     path_parameters = base_dir / simulation_name / "rejection_sampling" / f"sim_{i + 1}"
        #     parameters_sim = simulate_parameters(model, meta_parameters, fix_weights=True)


        # dummy_feature_counts = {
        #    'clusters': np.zeros((model.shapes.n_clusters,
        #                          model.shapes.n_features))
        # } | {
        #    conf: np.zeros((n_groups,
        #                    model.shapes.n_features))
        #    for conf, n_groups in model.shapes.n_groups.items()
        # }
        #
        # sample = Sample.from_numpy_arrays(
        #     clusters=parameters_sim['clusters'],
        #     weights=parameters_sim['weights'],
        #     confounders=parameters_sim['confounders'],
        #     source=parameters_sim['source'],
        #     feature_counts=dummy_feature_counts,
        #     model_shapes=model.shapes,
        # )
        # print(sample)

    # Export the priors in a format sBayes can interpret
    # Priors: Numbers not strings

    # Run sBayes on time use plus
    # Read prior and test
    # Concentration for prior in sBayes?
    # initial_counts to 0
    # Variance for Gaussian prior: simulate from Jeffrey's prior? use a constant? How to handle inference?
    # Fix poisson
    # Fix export meta categorical

def cli():
    parser = argparse.ArgumentParser(description="sBayes simulation script")
    parser.add_argument("name", type=str, help="Name for the simulation run")
    parser.add_argument("config", type=Path, help="Path to a sBayes config file used to extract the prior settings for the simulation")
    parser.add_argument("n_sim", type=int, default=10, help="Number of simulations to run")
    args = parser.parse_args()
    main(
        name=args.name,
        config=args.config,
        n_sim=args.n_sim,
    )


if __name__ == '__main__':
    main('test_gaussian', Path('experiments/simulation/test_gaussian/config.yaml'), 100)
    # main('test_categorical', Path('experiments/simulation/test_categorical/config.yaml'), 100)
