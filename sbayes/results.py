import typing as tp
from itertools import permutations

import numpy as np
import numpy.typing as npt
import pandas as pd

from sbayes.util import parse_cluster_columns


T = tp.TypeVar("T", bound="Results")


class Results:

    """
    Class for reading, storing, summarizing results of a sBayes analysis.

    Attributes:
        areas (npt.NDArray[bool]): Array containing samples of areas.
            shape: (n_areas, n_samples, n_sites)
        parameters (pd.DataFrame): Data-frame containing sample information about parameters
                                   and likelihood, prior and posterior probabilities.
    """

    def __init__(self, areas: npt.NDArray[bool], parameters: pd.DataFrame):
        self.areas = areas
        self.parameters = parameters

        # Cached properties
        self._inheritance = None

        # Parse feature, state, family and area names
        self.feature_names, self.feature_states = extract_features_and_states(parameters)
        self.family_names = self.get_family_names(parameters.columns)
        self.area_names = self.get_area_names(parameters.columns)

        # Parse parameters from the parameters dataframe

        # The sample index
        self.sample_id = self.parameters["Sample"].to_numpy(dtype=int)

        # Model parameters
        # self.weights = self.parse_weights(self.parameters)
        # self.alpha = self.parse_universal_probs(self.parameters)
        # self.beta = self.parse_family_probs(self.parameters)
        # self.gamma = self.parse_area_probs(self.parameters)
        self.weights = Results.read_dictionary(self.parameters, 'w_')
        self.alpha = Results.read_dictionary(self.parameters, 'alpha_')
        if self.inheritance:
            self.beta = Results.read_dictionary(self.parameters, 'beta_')
        self.gamma = Results.read_dictionary(self.parameters, 'gamma_')

        # Posterior, likelihood, prior
        self.posterior = self.parameters["posterior"].to_numpy(dtype=float)
        self.likelihood = self.parameters["likelihood"].to_numpy(dtype=float)
        self.prior = self.parameters["prior"].to_numpy(dtype=float)

        # Posterior, likelihood, prior contribution per area
        self.posterior_single_areas = Results.read_dictionary(self.parameters, 'post_')
        self.likelihood_single_areas = Results.read_dictionary(self.parameters, 'lh_')
        self.prior_single_areas = Results.read_dictionary(self.parameters, 'prior_')

    @property
    def inheritance(self):
        if self._inheritance is None:
            self._inheritance = any(k.startswith('gamma_') for k in self.parameters.columns)
        return self._inheritance

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    def n_areas(self):
        return self.areas.shape[0]

    @property
    def n_samples(self):
        return self.areas.shape[1]

    @property
    def n_sites(self):
        return self.areas.shape[2]

    def __getitem__(self, item):
        if item in [
            "feature_names",
            "sample_id",
            "areas",
            "weights",
            "alpha",
            "beta",
            "gamma",
            "posterior",
            "likelihood",
            "prior",
            "posterior_single_areas",
            "likelihood_single_areas",
            "prior_single_areas",
        ]:
            return getattr(self, item)
        else:
            raise ValueError(f"Unknown parameter name ´{item}´")

    @classmethod
    def from_csv_files(cls: tp.Type[T], areas_path: str, parameters_path: str) -> T:
        areas = cls.read_areas(areas_path)
        parameters = cls.read_stats(parameters_path)
        return cls(areas, parameters)

    @staticmethod
    def read_areas(txt_path: str) -> npt.NDArray[bool]:
        areas_list = []
        with open(txt_path, "r") as f_sample:
            # This makes len(result) = number of areas (flipped array)

            # Split the sample
            # len(byte_results) equals the number of samples
            byte_results = (f_sample.read()).split("\n")

            # Get the number of areas
            n_areas = len(byte_results[0].split("\t"))

            # Append empty arrays to result, so that len(result) = n_areas
            for i in range(n_areas):
                areas_list.append([])

            # Process each sample
            for sample in byte_results:
                if len(sample) == 0:
                    continue

                # Parse each sample
                parsed_sample = parse_cluster_columns(sample)
                # shape: (n_areas, n_sites)

                # Add each item in parsed_area_columns to the corresponding array in result
                for j in range(len(parsed_sample)):
                    areas_list[j].append(parsed_sample[j])

        return np.array(areas_list, dtype=bool)

    @staticmethod
    def read_stats(txt_path: str) -> pd.DataFrame:
        """Read stats for results files (<experiment_path>/stats_<scenario>.txt).

        Args:
            txt_path: path to results file
        """
        return pd.read_csv(txt_path, delimiter="\t")

    @staticmethod
    def read_dictionary(dataframe, search_key):
        """Helper function used for reading parameter dicts from pandas data-frame."""
        param_dict = {}
        for column_name in dataframe.columns:
            if column_name.startswith(search_key):
                param_dict[column_name] = dataframe[column_name].to_numpy(
                    dtype=np.float
                )

        return param_dict

    def parse_weights(self, parameters: pd.DataFrame) -> tp.Dict[str, npt.NDArray]:
        inheritance = f"w_inheritance_{self.feature_names[0]}" in parameters.columns
        weights = {}
        for f in self.feature_names:
            w_univ = parameters[f"w_universal_{f}"].to_numpy(dtype=np.float)
            w_cont = parameters[f"w_contact_{f}"].to_numpy(dtype=np.float)
            if inheritance:
                w_inhe = parameters[f"w_inheritance_{f}"].to_numpy(dtype=np.float)
                weights[f] = np.array([w_univ, w_cont, w_inhe])
            else:
                weights[f] = np.array([w_univ, w_cont])

        return weights

    def parse_probs(
        self,
        parameters: pd.DataFrame,
        param_name: str,
    ) -> tp.Dict[str, npt.NDArray]:

        param = {}
        for i_f, f in enumerate(self.feature_names):
            param[f] = np.array(
                [parameters[f"{param_name}_{f}_{s}"] for s in self.feature_states[i_f]]
            )

        f0 = self.feature_names[0]
        assert param[f0].shape == (len(self.feature_states[0]), self.n_samples)
        assert len(param) == self.n_features

        return param

    def parse_universal_probs(self, parameters: pd.DataFrame) -> tp.Dict:
        return self.parse_probs(parameters, 'alpha')

    def parse_family_probs(self, parameters: pd.DataFrame) -> tp.Dict:
        beta = {}
        for fam in self.family_names:
            beta[fam] = self.parse_probs(self.parameters, f'beta_{fam}')
        return beta

    def parse_area_probs(self) -> tp.Dict:
        gamma = {}
        for area in self.area_names:
            gamma[area] = self.parse_probs(self.parameters, f'gamma_{area}')
        return gamma

    @staticmethod
    def get_family_names(column_names) -> tp.List[str]:
        family_names = []
        for key in column_names:
            if not key.startswith('beta_'):
                continue
            _, fam, _, _ = key.split('_')
            if fam not in family_names:
                family_names.append(fam)
        return family_names

    @staticmethod
    def get_area_names(column_names) -> tp.List[str]:
        area_names = []
        for key in column_names:
            if not key.startswith('gamma_'):
                continue
            _, area, _ = key.split('_', maxsplit=2)
            if area not in area_names:
                area_names.append(area)
        return area_names

    # def rank_areas(self):
    #     post_per_area = self.likelihood_single_areas
    #     to_rank = np.mean(post_per_area, axis=0)
    #     ranked = np.argsort(-to_rank)
    #
    #     # probability per area in log-space
    #     # p_total = logsumexp(to_rank)
    #     # p = to_rank[np.argsort(-to_rank)] - p_total
    #
    #     ranked_areas = []
    #     ranked_lh = []
    #     ranked_prior = []
    #     ranked_posterior = []
    #     ranked_p_areas = []
    #
    #     print("Ranking areas ...")
    #     for s in range(len(samples["sample_zones"])):
    #         ranked_areas.append(samples["sample_zones"][s][ranked])
    #     samples["sample_zones"] = ranked_areas
    #
    #     print("Ranking lh areas ...")
    #     for s in range(len(samples["sample_lh_single_zones"])):
    #         ranked_lh.append([samples["sample_lh_single_zones"][s][r] for r in ranked])
    #     samples["sample_lh_single_zones"] = ranked_lh
    #
    #     print("Ranking prior areas ...")
    #     for s in range(len(samples["sample_prior_single_zones"])):
    #         ranked_prior.append(
    #             [samples["sample_prior_single_zones"][s][r] for r in ranked]
    #         )
    #     samples["sample_prior_single_zones"] = ranked_prior
    #
    #     print("Ranking posterior areas ...")
    #     for s in range(len(samples["sample_posterior_single_zones"])):
    #         ranked_posterior.append(
    #             [samples["sample_posterior_single_zones"][s][r] for r in ranked]
    #         )
    #     samples["sample_posterior_single_zones"] = ranked_posterior
    #
    #     print("Ranking p areas ...")
    #     for s in range(len(samples["sample_p_zones"])):
    #         ranked_p_areas.append(samples["sample_p_zones"][s][ranked])
    #     samples["sample_p_zones"] = ranked_p_areas
    #
    #     return samples
    #
    # def match_areas(self):
    #     n_areas, n_samples, n_sites = self.areas.shape
    #
    #     # n_samples, n_sites, n_areas = area_samples.shape
    #     s_sum = np.zeros((n_sites, n_areas))
    #
    #     # All potential permutations of cluster labels
    #     perm = list(permutations(range(n_areas)))
    #     matching_list = []
    #     for i_sample in range(n_samples):
    #         s = self.areas[:, i_sample, :].T
    #
    #         def clustering_agreement(p):
    #             """How many sites match the previous sample for permutation `p`?"""
    #             return np.sum(s_sum * s[:, p])
    #
    #         best_match = max(perm, key=clustering_agreement)
    #         matching_list.append(list(best_match))
    #         s_sum += s[:, best_match]
    #
    #     # Reorder chains according to matching
    #     reordered_zones = []
    #     reordered_p_zones = []
    #     reordered_lh = []
    #     reordered_prior = []
    #     reordered_posterior = []
    #
    #     print("Matching areas ...")
    #     for s in range(n_samples):
    #         self.areas[:, s, :] = self.areas[matching_list[s], s, :]
    #         x = self.alpha
    #         reordered_p_zones.append(samples["sample_p_zones"][s][matching_list[s]])
    #         reordered_lh.append(
    #             [samples["sample_lh_single_zones"][s][i] for i in matching_list[s]]
    #         )
    #         reordered_prior.append(
    #             [samples["sample_prior_single_zones"][s][i] for i in matching_list[s]]
    #         )
    #         reordered_posterior.append(
    #             [
    #                 samples["sample_posterior_single_zones"][s][i]
    #                 for i in matching_list[s]
    #             ]
    #         )
    #
    #     return samples


def extract_features_and_states(parameters: pd.DataFrame):
    feature_names = []
    state_names = []

    # We look at all ´alpha´ columns, since they contain each feature-state exactly once.
    alpha_columns = [c for c in parameters.columns if c.startswith("alpha_")]

    for c in alpha_columns:
        # Column name format is 'alpha_{featurename}_{statename}'
        f, _, s = c[6:].rpartition('_')

        # Add the feature name to the list (if not present)
        if f not in feature_names:
            feature_names.append(f)
            state_names.append([])

        # Find the index of feature f
        i_f = feature_names.index(f)

        # Add state s to the state_names list of feature f
        state_names[i_f].append(s)

    return feature_names, state_names
