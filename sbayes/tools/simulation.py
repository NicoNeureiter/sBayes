import pandas as pd
import numpy as np
import os
import random
import yaml
import string
from sbayes.load_data import Objects, Confounder
from collections import OrderedDict
from sbayes.model.likelihood import normalize_weights
from sbayes.preprocessing import sample_categorical
from scipy.stats import invgamma, gamma
from sbayes.util import set_experiment_name, fix_relative_path, format_cluster_columns


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


def simulate_objects(obj_meta: dict):
    """ Simulate objects with attributes location (x,y) and ID
        Args:
            obj_meta (dict): meta information about the simulated objects
        Returns:
            (Objects): the simulated objects"""
    n = obj_meta['n']
    # Simulate n objects with random locations
    df = pd.DataFrame().assign(
        x=np.random.rand(n),
        y=np.random.rand(n),
        id=range(n),
        name=["".join(['object', str(i)]) for i in range(n)])

    return Objects.from_dataframe(df)


def simulate_confounder_assignment(ob: Objects, conf_meta: dict):
    """ Randomly assign objects to confounders
        Args:
            ob (Objects): simulated objects with attributes location (x,y) and ID
            conf_meta (dict): confounder names, their applicable groups and relative frequency
        Returns:
            (OrderedDict): the simulated assignment of objects to confounders"""

    df = pd.DataFrame()

    for cf, g in conf_meta.items():
        groups = list(g.keys())
        p = list(g.values())
        df[cf] = np.random.choice(groups, size=ob.n_objects, p=p)

    conf = OrderedDict()
    for c in df.columns.tolist():
        conf[c] = Confounder.from_dataframe(data=df, confounder_name=c)

    return conf


def simulate_cluster_assignment(ob: Objects, clust_meta: dict):
    """ Randomly assign objects to mutually exclusive clusters
        Args:
            ob (Objects): simulated objects with attributes location (x,y) and ID
            clust_meta (dict): clusters and their relative frequencies
        Returns:
            (np.ndarray):  the simulated assignment of objects to clusters"""

    # Randomly assign the objects to three mutually exclusive clusters
    c = list(clust_meta.keys())
    p = list(clust_meta.values())

    df = pd.DataFrame()
    df['clust'] = np.random.choice(c + ["0"], size=ob.n_objects, p=p + [1 - sum(p)])
    clust_bin = np.zeros((len(c), ob.n_objects), dtype=bool)

    for i, cl in enumerate(c):
        clust_bin[i] = df['clust'].apply(lambda x: True if x == cl else False).values
    return clust_bin


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


def simulate_cluster_effect_prior(feat: dict,  clust: dict, p_hyper: dict):
    """ Simulate the prior distributions for the cluster effect, explicitly writing it out for each feature
        Args:
            feat (dict): number of features per feature type
            clust (dict): the clusters
            p_hyper (dict): probabilities of the hyperparameter values for simulating the cluster effect prior
        Returns:
            (dict): cluster effect prior for each feature, structured per feature type """

    prior = {}
    for ft, fs in feat.items():
        prior[ft] = {}
        for cl in clust.keys():
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
                prior[ft][cl] = dict(mean=mean,
                                     variance=variance)
            elif ft == 'poisson':
                rate = dict(alpha_0=np.random.uniform(p_hyper[ft]['rate']['alpha_0'][0],
                                                      p_hyper[ft]['rate']['alpha_0'][1], size=fs),
                            beta_0=np.random.uniform(p_hyper[ft]['rate']['beta_0'][0],
                                                     p_hyper[ft]['rate']['beta_0'][1], size=fs))
                prior[ft][cl] = dict(rate=rate)
            else:
                raise ValueError("Feature type nor supported")
    return prior


def simulate_weights(prior: dict):
    """ Simulates weights of the cluster and confounding effects for all features
    Args:
        prior (dict): The Dirichlet weights prior for each feature, structured per feature type

    Returns:
        (dict): simulated weights for each effect and each feature, structured per feature type"""

    weights = {}
    for ft, fs in prior.items():
        w = []
        for f in fs['concentration']:
            w.append(np.random.dirichlet(f))
        weights[ft] = np.vstack(w)

    return weights


def simulate_confounding_effect(prior: dict):
    """ Simulates the confounding effect per confounder for all features
        Args:
            prior (dict): The confounding effect prior for each feature, structured per feature type

        Returns:
            (dict): simulated confounding effect per confounder for each feature, structured per feature type"""

    conf_effect = {}
    for ft, conf in prior.items():
        conf_effect[ft] = {}
        for c, groups in conf.items():
            conf_effect[ft][c] = {}
            for g, params in groups.items():
                conf_effect[ft][c][g] = {}
                if ft == 'categorical':
                    max_categorical_states = \
                        max(len(i) for i in next(iter(next(iter(prior['categorical'].values())).
                                                      values()))['concentration'])
                    p = []
                    for f in params['concentration']:
                        p_f = np.zeros(max_categorical_states)
                        p_f[:len(f)] = np.random.dirichlet(f, size=1)
                        p.append(p_f)
                    conf_effect[ft][c][g] = np.vstack(p)
                elif ft == 'gaussian':
                    mean = np.random.normal(params['mean']['mu_0'], params['mean']['sigma_0'])
                    variance = invgamma.rvs(params['variance']['alpha_0'], params['variance']['beta_0'])
                    conf_effect[ft][c][g] = dict(mean=mean,
                                                 variance=variance)
                elif ft == 'poisson':
                    rate = gamma.rvs(params['rate']['alpha_0'], params['rate']['beta_0'])
                    conf_effect[ft][c][g] = dict(rate=rate)
                else:
                    raise ValueError("Feature type not supported")

    return conf_effect


def simulate_cluster_effect(prior: dict):
    """ Simulates the cluster effect for all features
        Args:
            prior (dict): The cluster effect prior for each feature, structured per feature type

        Returns:
            (dict): simulated cluster effect for each feature, structured per feature type"""

    clust_effect = {}
    for ft, clust in prior.items():
        clust_effect[ft] = {}
        for cl, params in clust.items():

            if ft == 'categorical':
                max_categorical_states = max(len(i) for i in next(iter(prior['categorical'].values()))['concentration'])
                p = []
                for f in params['concentration']:
                    p_f = np.zeros(max_categorical_states)
                    p_f[: len(f)] = np.random.dirichlet(f, size=1)
                    p.append(p_f)
                clust_effect[ft][cl] = np.vstack(p)

            elif ft == 'gaussian':
                mean = np.random.normal(params['mean']['mu_0'], params['mean']['sigma_0'])
                variance = invgamma.rvs(params['variance']['alpha_0'], params['variance']['beta_0'])
                clust_effect[ft][cl] = dict(mean=mean,
                                            variance=variance)

            elif ft == 'poisson':
                rate = gamma.rvs(params['rate']['alpha_0'], params['rate']['beta_0'])
                clust_effect[ft][cl] = dict(rate=rate)
            else:
                raise ValueError("Feature type not supported")

    return clust_effect


def simulate_features(weights: dict, conf_effect: dict, clust_effect: dict,
                      conf: dict, clust: np.ndarray, names: dict):
    """Simulate features.
        Args:
            weights (dict): the influence of the confounding and cluster effects for each feature
            conf_effect (dict): the confounding effect per confounder for each feature
            clust_effect (dict): the cluster effect for each feature
            conf (dict): the assignments of sites to groups of a confounder
            clust (np.ndarray): the assignment of sites to clusters
                shape: (n_clusters, n_objects)
            names (dict): The feature and state names

        Returns:
            (dict): The simulated features per feature type"""

    # Which object is assigned to which effect?
    n_objects = clust.shape[1]
    effect_assignment = [np.any(clust, axis=0)]
    assignment_order = {"cluster": 0}

    for i, (n, c) in enumerate(conf.items()):
        effect_assignment.append(np.any(c['group_assignment'], axis=0))
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
                lh_p = normed_weights[f, :, assignment_order['cluster']] * lh_p_cl
                for n, c in conf.items():
                    p_c = np.array(list(conf_effect[ft][n].values()))[:, f, :]
                    lh_p_c = c['group_assignment'].T.dot(p_c).T
                    lh_p += normed_weights[f, :, assignment_order[n]] * lh_p_c
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

                lh_mean = normed_weights[f, :, assignment_order['cluster']] * lh_mean_cl
                lh_variance = normed_weights[f, :, assignment_order['cluster']] * lh_variance_cl

                for n, c in conf.items():
                    mean_c = np.array([g['mean'] for g in conf_effect[ft][n].values()])[:, f]
                    variance_c = np.array([g['variance'] for g in conf_effect[ft][n].values()])[:, f]

                    lh_mean_c = c['group_assignment'].T.dot(mean_c).T
                    lh_variance_c = c['group_assignment'].T.dot(variance_c).T
                    lh_mean += normed_weights[f, :, assignment_order[n]] * lh_mean_c
                    lh_variance += normed_weights[f, :, assignment_order[n]] * lh_variance_c

                feat[ft][:, f] = np.random.normal(lh_mean, lh_variance)

        elif ft == 'poisson':

            feat[ft] = np.zeros((n_objects, n_feat), dtype=int)

            for f, f_name in enumerate(names[ft]):

                rate_cl = np.array([cl['rate'] for cl in clust_effect[ft].values()])[:, f]
                # Compute the feature likelihood matrix (for all objects)
                lh_rate_cl = clust.T.dot(rate_cl).T

                lh_rate = normed_weights[f, :, assignment_order['cluster']] * lh_rate_cl

                for n, c in conf.items():
                    rate_c = np.array([g['rate'] for g in conf_effect[ft][n].values()])[:, f]
                    lh_rate_c = c['group_assignment'].T.dot(rate_c).T

                    lh_rate += normed_weights[f, :, assignment_order[n]] * lh_rate_c

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
                path (str): path for storing the simulation
            Returns:
                (str: the simulation name): """

    folder = fix_relative_path(path, os.getcwd()) or os.getcwd()
    os.makedirs(folder, exist_ok=True)
    write_priors(priors, names, folder)

    return sim_name


def export_parameters(params=None, names=None, path=None):
    """ Export the simulated data and parameters
        Args:
            params (dict): the parameters to simulate the data
            names (dict): simulated names of the features and states
            path (str): folder path for storing the simulation
            """

    folder = fix_relative_path(path, os.getcwd()) or os.getcwd()
    os.makedirs(folder, exist_ok=True)

    # Export the parameters as stats_sim.txt
    write_parameters(params=params,
                     names=names,
                     path=folder / 'stats_sim.txt')

    # Export the clusters as cluster_sim.txt
    with open(folder / 'clusters_sim.txt', "w") as file:
        file.write(format_cluster_columns(params['clusters']))
        file.close()


def export_meta(meta=dict, path=None):
    """ Export the simulated data and parameters
            Args:
                meta (DataFrame): the simulated the data
                path (str): folder path for storing the simulation
                """
    folder = fix_relative_path(path, os.getcwd()) or os.getcwd()
    os.makedirs(folder, exist_ok=True)

    with open(folder / 'meta_sim.yaml', 'w') as file:
        yaml.safe_dump(meta, file, sort_keys=False)


def export_data(data=None, path=None):
    """ Export the simulated data and parameters
        Args:
            data (DataFrame): the simulated the data
            path (str): folder path for storing the simulation
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
                path (str): path for saving the text file
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
                   folder (path): path for saving the priors
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

    # Cluster effect prior
    os.makedirs(folder / "cluster_effect", exist_ok=True)
    for ft, clust in priors['cluster_effect_prior'].items():
        for cl, cl_eff in clust.items():
            prior_yaml = {}
            for i_f, f in enumerate(feat_names[ft]):

                if ft == 'categorical':

                    prior_yaml[f] = \
                        dict(concentration={n: float_style % c for n, c in zip(names['categorical'][f],
                                                                               cl_eff['concentration'][i_f])})
                elif ft == 'gaussian':
                    prior_yaml[f] = dict(mean=dict(mu_0=float_style % cl_eff['mean']['mu_0'][i_f],
                                                   sigma_0=float_style % cl_eff['mean']['sigma_0'][i_f]))
                elif ft == 'poisson':
                    prior_yaml[f] = dict(rate=dict(alpha_0=float_style % cl_eff['rate']['alpha_0'][i_f],
                                                   beta_0=float_style % cl_eff['rate']['beta_0'][i_f]))

            path = folder / "_".join(['cluster_effect/cluster_effect', cl, ft, ext])
            with open(path, 'w') as file:
                yaml.safe_dump(prior_yaml, file, sort_keys=False)

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

                path = folder / "_".join(["confounding_effects/", conf, g, ft, ext])
                with open(path, 'w') as file:
                    yaml.safe_dump(prior_yaml, file, sort_keys=False)


def simulate_prior(meta, hyper, path):
    """Simulate the priors and write to file.
            Args:
                meta (dict): meta information about the objects, clusters, features and confounding effects
                hyper(dict): hyperparameters for defining the priors
                path(str): file path to store the results

            Returns:
                (dict): the simulated prior"""

    # Simulate hyperparameters to define the weights prior
    weights_prior_sim = simulate_weights_prior(meta['features'], meta['confounders'], hyper['weights'])

    # Simulate hyperparameters to define the confounding effect prior
    conf_prior_sim = simulate_confounding_effect_prior(meta['features'], meta['confounders'],
                                                       hyper['confounding_effects'])

    # Simulate hyperparameters to define the cluster effect prior
    clust_prior_sim = simulate_cluster_effect_prior(meta['features'], meta['clusters'],
                                                    hyper['cluster_effect'])

    export_priors(priors=dict(weights_prior=weights_prior_sim,
                              confounding_effect_prior=conf_prior_sim,
                              cluster_effect_prior=clust_prior_sim),
                  names=meta['names'], path=path)

    return dict(weights=weights_prior_sim,
                confounding_effects=conf_prior_sim,
                cluster_effect=clust_prior_sim)


def simulate_parameters(prior_sim, meta, path):
    """Simulate the parameters and write to file
            Args:
                prior_sim (dict): simulated prior
                meta(dict): meta information about the objects, clusters, features and confounding effects
                path(str): file path to store the results
            Returns:
                (dict): the simulated parameters"""

    # Define the objects
    obj_sim = simulate_objects(meta['objects'])

    # Set the assignment to confounders
    conf_sim = simulate_confounder_assignment(obj_sim, meta['confounders'])

    # Set the assignment to clusters
    clust_sim = simulate_cluster_assignment(obj_sim, meta['clusters'])

    # Simulate the weights
    weights_sim = simulate_weights(prior_sim['weights'])

    # Simulate the confounding effect
    conf_effects_sim = simulate_confounding_effect(prior_sim['confounding_effects'])

    # Simulate the cluster effect
    clust_effect_sim = simulate_cluster_effect(prior_sim['cluster_effect'])

    export_parameters(params=dict(weights=weights_sim,
                                  clusters=clust_sim,
                                  confounders=conf_sim,
                                  features=meta['features'],
                                  cluster_effects=clust_effect_sim,
                                  confounding_effects=conf_effects_sim),
                      names=meta['names'],
                      path=path)

    return dict(objects=obj_sim, confounders=conf_sim, clusters=clust_sim,
                weights=weights_sim, confounding_effects=conf_effects_sim,
                cluster_effect=clust_effect_sim)


def simulate_data(param_sim, meta, path):
    """Simulate the parameters and write to file
                Args:
                    param_sim (dict): simulated parameters
                    meta(dict): meta information about the objects, clusters, features and confounding effects
                    path(str): file path to store the results
                Returns:
                    (dict): the simulated data"""

    # Simulate the features
    features_sim = simulate_features(param_sim['weights'],
                                     param_sim['confounding_effects'],
                                     param_sim['cluster_effect'],
                                     param_sim['confounders'],
                                     param_sim['clusters'],
                                     meta['names'])

    # Combine the simulated objects, confounders and features in a DataFrame
    data_sim, feature_names_sim = combine_data(param_sim['objects'],
                                               param_sim['confounders'],
                                               features_sim)

    # Export the simulated parameters and data
    export_data(data=data_sim, path=path)


objects_meta = dict(n=100)

# Names of simulated confounders and simulated groups per confounder and their relative frequency (must sum to 1).
confounders_meta = dict(
    conf_1=dict(a=0.3, b=0.4, c=0.3),
    conf_2=dict(d=0.5, e=0.5)
)

# Names of simulated clusters and their relative frequency (can but should not sum to 1).
clusters_meta = dict(
    cluster_1=0.1,
    cluster_2=0.05,
    cluster_3=0.2
)

# Number of simulated features per feature type
# For categorical features the tuple (2, 3) simulates 3 features with 2 states
features_meta = dict(
    #categorical=[(2, 10), (3, 4), (4, 5)],
    gaussian=10,
#    poisson=10,
)

# features_meta = dict(gaussian=20)


# Range of the concentration values for simulating the weights prior
# A higher concentration yields a pointier prior
hyper_parameters_weights_prior = dict(concentration=(1, 1))

# Range of the hyperparameter values for simulating the confounding effect prior
hyper_parameters_confounding_effect_prior = dict(
    categorical=dict(
        concentration=(1, 1)),
    gaussian=dict(
        mean=dict(
            mu_0=(0, 0),
            sigma_0=(1000, 1000)),
        variance=dict(
            alpha_0=(1, 1),
            beta_0=(1, 1))),
    poisson=dict(
        rate=dict(
            alpha_0=(0, 0),
            beta_0=(10, 10))))

# Range of the hyperparameter values for simulating the cluster effect prior
hyper_parameters_cluster_effect_prior = dict(
    categorical=dict(
        concentration=(1, 1)),
    gaussian=dict(
        mean=dict(
            mu_0=(0, 0),
            sigma_0=(1000, 1000)),
        variance=dict(
            alpha_0=(1, 1),
            beta_0=(1, 1))),
    poisson=dict(
        rate=dict(
            alpha_0=(0, 0),
            beta_0=(10, 10))))

meta_parameters = dict(objects=objects_meta,
                       confounders=confounders_meta,
                       clusters=clusters_meta,
                       features=features_meta,
                       names=define_names(features_meta))

hyper_parameters = dict(weights=hyper_parameters_weights_prior,
                        confounding_effects=hyper_parameters_confounding_effect_prior,
                        cluster_effect=hyper_parameters_cluster_effect_prior)

folder_path = "../../experiments/simulation"
simulation_name = set_simulation_name(features_meta)

# Simulate the prior
path_prior = "/".join([folder_path, simulation_name, "prior"])
prior_probability_sim = simulate_prior(meta_parameters, hyper_parameters, path_prior)

path_meta = "/".join([folder_path, simulation_name, "meta"])
export_meta(meta=meta_parameters, path=path_meta)

n_simulation = 10

for i in range(n_simulation):
    # Simulate the parameters
    path_parameters = "/".join([folder_path, simulation_name, "parameters",  "sim_"+str(i+1)])
    parameters_sim = simulate_parameters(prior_probability_sim, meta_parameters, path_parameters)

    # Simulate the data
    path_data = "/".join([folder_path, simulation_name, "data", "sim_"+str(i+1)])
    simulate_data(parameters_sim, meta_parameters, path_data)


# Export the priors in a format sBayes can interpret
# Priors: Numbers not strings


# Run sBayes on time use plus 
# Read prior and test
# Concentration for prior in sBayes?
# initial_counts to 0
# Variance for Gaussian prior: simulate from Jeffrey's prior? use a constant? How to handle inference?
# Fix poisson
# Fix export meta categorical