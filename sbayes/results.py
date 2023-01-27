from __future__ import annotations
from typing import Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from sbayes.util import PathLike, parse_cluster_columns


TResults = TypeVar("TResults", bound="Results")


class Results:

    """
    Class for reading, storing, summarizing results of a sBayes analysis.

    Attributes:
        clusters (NDArray[bool]): Array containing samples of clusters.
            shape: (n_clusters, n_samples, n_sites)
        parameters (pd.DataFrame): Data-frame containing sample information about parameters
                                   and likelihood, prior and posterior probabilities.
        groups_by_confounders (dict[str, list[str]): A list of groups for each confounder.
    """

    def __init__(
        self,
        clusters: NDArray[bool],
        parameters: pd.DataFrame,
        burn_in: float = 0.1,
    ):
        clusters, parameters = self.drop_burnin(clusters, parameters, burn_in)
        self.clusters = clusters
        self.parameters = parameters

        self.groups_by_confounders = self.get_groups_by_confounder(parameters.columns)
        self.cluster_names = self.get_cluster_names(parameters.columns)

        # Parse feature, state, family and area names
        self.feature_names = extract_feature_names(parameters)
        self.feature_states = [
            extract_state_names(parameters, prefix=f"areal_{self.cluster_names[0]}_{f}_")
            for f in self.feature_names
        ]

        # The sample index
        self.sample_id = self.parameters["Sample"].to_numpy(dtype=int)

        # Model parameters
        self.weights = self.parse_weights(self.parameters)
        self.areal_effect = self.parse_areal_effect(self.parameters)
        self.confounding_effects = self.parse_confounding_effects(self.parameters)
        # self.weights = Results.read_dictionary(self.parameters, "w_")
        # self.areal_effect = Results.read_dictionary(self.parameters, "areal_")
        # self.confounding_effects = {
        #     conf: Results.read_dictionary(self.parameters, f"{conf}_")
        #     for conf in self.groups_by_confounders
        # }

        # Posterior, likelihood, prior
        self.posterior = self.parameters["posterior"].to_numpy(dtype=float)
        self.likelihood = self.parameters["likelihood"].to_numpy(dtype=float)
        self.prior = self.parameters["prior"].to_numpy(dtype=float)

        # Posterior, likelihood, prior contribution per area
        self.posterior_single_clusters = Results.read_dictionary(
            self.parameters, "post_"
        )
        self.likelihood_single_clusters = Results.read_dictionary(
            self.parameters, "lh_"
        )
        self.prior_single_clusters = Results.read_dictionary(self.parameters, "prior_")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[0]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[1]

    @property
    def n_objects(self) -> int:
        return self.clusters.shape[2]

    @property
    def confounders(self) -> list[str]:
        return list(self.groups_by_confounders.keys())

    @property
    def n_confounders(self) -> int:
        return len(self.groups_by_confounders)

    def __getitem__(self, item: str):
        if item in [
            "feature_names",
            "sample_id",
            "clusters",
            "weights",
            "alpha",
            "beta",
            "gamma",
            "posterior",
            "likelihood",
            "prior",
            "posterior_single_clusters",
            "likelihood_single_clusters",
            "prior_single_clusters",
        ]:
            return getattr(self, item)
        else:
            raise ValueError(f"Unknown parameter name ´{item}´")

    @classmethod
    def from_csv_files(
        cls: type[TResults],
        clusters_path: PathLike,
        parameters_path: PathLike,
        burn_in: float = 0.1
    ) -> TResults:
        clusters = cls.read_clusters(clusters_path)
        parameters = cls.read_stats(parameters_path)
        return cls(clusters, parameters, burn_in=burn_in)

    @staticmethod
    def drop_burnin(clusters, parameters, burn_in):
        # Translate burn_in fraction to index
        n_total_samples = clusters.shape[1]
        burn_in_index = int(burn_in * n_total_samples)

        # Drop burnin samples from both arrays
        clusters = clusters[:, burn_in_index:, :]
        parameters = parameters.iloc[burn_in_index:]

        return clusters, parameters

    @staticmethod
    def read_clusters(txt_path: PathLike) -> NDArray[bool]:  # shape: (n_clusters, n_samples, n_sites)
        """Read the cluster samples from the text file at `txt_path` and return as a
        boolean numpy array."""
        clusters_list = []
        with open(txt_path, "r") as f_sample:
            # This makes len(result) = number of clusters (flipped array)

            # Split the sample
            # len(byte_results) equals the number of samples
            byte_results = (f_sample.read()).split("\n")

            # Get the number of clusters
            n_clusters = len(byte_results[0].split("\t"))

            # Append empty arrays to result, so that len(result) = n_clusters
            for i in range(n_clusters):
                clusters_list.append([])

            # Process each sample
            for sample in byte_results:
                if len(sample) == 0:
                    continue

                # Parse each sample
                parsed_sample = parse_cluster_columns(sample)
                # shape: (n_clusters, n_sites)

                # Add each item in parsed_area_columns to the corresponding array in result
                for j in range(len(parsed_sample)):
                    clusters_list[j].append(parsed_sample[j])

        return np.array(clusters_list, dtype=bool)

    @staticmethod
    def read_stats(txt_path: PathLike) -> pd.DataFrame:
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
                param_dict[column_name] = dataframe[column_name].to_numpy(dtype=float)

        return param_dict

    def parse_weights(self, parameters: pd.DataFrame) -> dict[str, NDArray]:
        """Parse weights array for each feature in a dictionary from the parameters
        data-frame.

        Args:
            parameters:

        Returns:
            dictionary mapping feature names to corresponding weights arrays. Each weights
                array has shape (n_samples, 1 + n_confounders).

        """
        # The components include the areal effect and all confounding effects and define
        # the dimensions of weights for each feature.
        components = ["areal"] + list(self.groups_by_confounders.keys())

        # Collect weights by feature
        weights = {}
        for f in self.feature_names:
            weights[f] = np.column_stack(
                [parameters[f"w_{c}_{f}"].to_numpy(dtype=float) for c in components]
            )

        return weights

    def parse_probs(
        self,
        parameters: pd.DataFrame,
        prefix: str,
    ) -> dict[str, NDArray[float]]:
        """Parse a categorical probabilities for each feature. The probabilities are
        specified in the columns starting with `prefix` in the `parameters` data-frame.

        Args:
            parameters: The data-frame of all logged parameters from a sbayes analysis.
            prefix: The prefix identifying the parameter to be parsed.

        Returns:
            The parsed dictionary mapping feature names to probability arrays.
                shape for each feature f: (n_states_f,)
        """

        param = {}
        for i_f, f in enumerate(self.feature_names):
            param[f] = np.column_stack(
                [parameters[f"{prefix}_{f}_{s}"] for s in self.feature_states[i_f]]
            )

        assert len(param) == self.n_features
        return param

    def parse_areal_effect(self, parameters: pd.DataFrame) -> dict[str, dict]:
        """Parse a categorical probabilities for each feature in each cluster. The
         probabilities are specified in the columns starting with `areal_` in the
         `parameters` data-frame.

        Args:
            parameters: The data-frame of all logged parameters from a sbayes analysis.

        Returns:
            Nested dictionary of form {cluster_name: {feature_name: probabilities}}.
                shape for each cluster and each feature f: (n_states_f,)
        """
        areal_effect = {
            cluster: self.parse_probs(parameters, f"areal_{cluster}")
            for cluster in self.cluster_names
        }
        return areal_effect

    def parse_confounding_effects(
        self, parameters: pd.DataFrame
    ) -> dict[str, dict]:
        """Parse a categorical probabilities for each feature in each confounder. The
         probabilities are specified in the `parameters` data-frame in columns starting
         with `{c}_` for a confounder c.

        Args:
            parameters: The data-frame of all logged parameters from a sbayes analysis.

        Returns:
            Nested dictionary of form {confounder_name: {group_name: {feature_name: probabilities}}}.
                shape for each cluster and each feature f: (n_states_f,)
        """
        conf_effects = {
            conf: {g: self.parse_probs(parameters, f"{conf}_{g}") for g in groups}
            for conf, groups in self.groups_by_confounders.items()
        }
        return conf_effects

    @staticmethod
    def get_family_names(column_names) -> list[str]:
        family_names = []
        for key in column_names:
            if not key.startswith("beta_"):
                continue
            _, fam, _, _ = key.split("_")
            if fam not in family_names:
                family_names.append(fam)
        return family_names

    @staticmethod
    def get_groups_by_confounder(
        column_names: Sequence[str],
    ) -> dict[str, list[str]]:
        """Create a dictionary containing all confounder names as keys and a list of
        corresponding group names as values. The dictionary is extracted from the column
        names in a csv file of logged sbayes parameters."""

        groups_by_confounder = {}

        # We use to weights columns to find confounder names
        for key in column_names:
            # Skip if not a weights column
            if not key.startswith("w_"):
                continue

            # Second part of key in weights columns defines the confounder name
            _, conf, _ = key.split("_", maxsplit=2)

            # Skip areal effects
            if conf == "areal":
                continue

            # Skip already added
            if conf in groups_by_confounder:
                continue

            # Otherwise, remember the confounder name and initialize the group name list
            groups_by_confounder[conf] = []

        # Collect the group names from the parameter columns of each confounder
        for conf in groups_by_confounder:
            for key in column_names:
                # Skip columns that are not on this confounder
                if not key.startswith(f"{conf}_"):
                    continue

                # Second part of key contains the group name
                _, group, _ = key.split("_", maxsplit=2)

                # Skip if already added
                if group in groups_by_confounder[conf]:
                    continue

                # Otherwise, remember the group name
                groups_by_confounder[conf].append(group)

        return groups_by_confounder

    @staticmethod
    def get_cluster_names(column_names) -> list[str]:
        area_names = []
        for key in column_names:
            if not key.startswith("areal_"):
                continue
            _, area, _ = key.split("_", maxsplit=2)
            if area not in area_names:
                area_names.append(area)
        return area_names

    def get_states_for_feature_name(self, f: str) -> list[str]:
        return self.feature_states[self.feature_names.index(f)]


def extract_features_and_states(
        parameters: pd.DataFrame,
        prefix: str
) -> (list[str], list[list[str]]):
    """Extract features names and state names of the given data-set.

    Args:
        parameters: The data-frame of all logged parameters from a sbayes analysis.
        prefix: The prefix identifying columns to be used.

    Returns:
        list of feature names and nested list of state names for each feature.
    """
    feature_names = []
    state_names = []

    # We look at all ´alpha´ columns, since they contain each feature-state exactly once.
    columns = [c for c in parameters.columns if c.startswith(f"{prefix}_")]

    for c in columns:
        # Column name format is '{prefix}_{featurename}_{statename}'
        f_s = c[len(prefix) + 1 :]
        f, _, s = f_s.partition("_")

        # Add the feature name to the list (if not present)
        if f not in feature_names:
            feature_names.append(f)
            state_names.append([])

        # Find the index of feature f
        i_f = feature_names.index(f)

        # Add state s to the state_names list of feature f
        state_names[i_f].append(s)

    return feature_names, state_names


def extract_feature_names(parameters: pd.DataFrame) -> list[str]:
    prefix = "w_areal_"
    feature_names = []
    for c in parameters.columns:
        if c.startswith(prefix):
            feature_names.append(c[len(prefix):])
    return feature_names


def extract_state_names(parameters: pd.DataFrame, prefix: str) -> list[str]:
    state_names = []
    for c in parameters.columns:
        if c.startswith(prefix):
            state_names.append(c[len(prefix):])
    return state_names
