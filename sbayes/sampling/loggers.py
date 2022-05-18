from typing import TextIO, Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import tables

from sbayes.load_data import Data
from sbayes.util import format_cluster_columns, get_best_permutation
from sbayes.model import Model

Sample = "sbayes.sampling.zone_sampling.Sample"


class ResultsLogger(ABC):
    def __init__(
        self,
        path: str,
        data: Data,
        model: Model,
    ):
        self.path: str = path
        self.data: Data = data
        self.model: Model = model.__copy__()

        self.file: Optional[TextIO] = None
        self.column_names: Optional[list] = None

    @abstractmethod
    def write_header(self, sample: Sample):
        pass

    @abstractmethod
    def _write_sample(self, sample: Sample):
        pass

    def write_sample(self, sample: Sample):
        if self.file is None:
            self.open()
            self.write_header(sample)
        self._write_sample(sample)

    def open(self):
        self.file = open(self.path, "w", buffering=1)
        # ´buffering=1´ activates line-buffering, i.e. flushing to file after each line

    def close(self):
        self.file.close()
        self.file = None


class ParametersCSVLogger(ResultsLogger):

    """The ParametersCSVLogger collects all real-valued parameters (weights, alpha, beta,
    gamma) and some statistics (cluster size, likelihood, prior, posterior) and continually
    writes them to a tab-separated text-file."""

    def __init__(
        self,
        *args,
        log_contribution_per_cluster: bool = True,
        float_format: str = "%.12g",
        match_clusters: bool = True,
    ):
        super().__init__(*args)
        self.float_format = float_format
        self.log_contribution_per_cluster = log_contribution_per_cluster
        self.match_clusters = match_clusters
        self.cluster_sum: Optional[npt.NDArray[int]] = None

    def write_header(self, sample):
        feature_names = self.data.features["names"]
        state_names = self.data.features["state_names"]
        column_names = ["Sample", "posterior", "likelihood", "prior"]

        # No need for matching if only 1 cluster (or no clusters at all)
        if sample.n_clusters <= 1:
            self.match_clusters = False

        # Initialize cluster_sum array for matching
        self.cluster_sum = np.zeros((sample.n_clusters, sample.n_sites), dtype=np.int)

        # Cluster sizes
        for i in range(sample.n_clusters):
            column_names.append(f"size_a{i}")

        # weights
        for i_feat, feat_name in enumerate(feature_names):
            column_names.append(f"w_universal_{feat_name}")
            column_names.append(f"w_contact_{feat_name}")
            if sample.inheritance:
                column_names.append(f"w_inheritance_{feat_name}")

        # alpha
        for i_feat, feat_name in enumerate(feature_names):
            for i_state, state_name in enumerate(state_names[i_feat]):
                col_name = f"alpha_{feat_name}_{state_name}"
                column_names.append(col_name)

        # gamma
        for a in range(sample.n_clusters):
            for i_feat, feat_name in enumerate(feature_names):
                for i_state, state_name in enumerate(state_names[i_feat]):
                    col_name = f"gamma_a{(a + 1)}_{feat_name}_{state_name}"
                    column_names.append(col_name)

        # beta
        for k, v in self.data.confounders.items():
            for group in :
                for f in feature_names:
                    for s in state_names[f]:
                        col_name = f"beta_{group}_{f}_{s}"
                        column_names += [col_name]

        # lh, prior, posteriors
        if self.log_contribution_per_cluster:
            for i in range(sample.n_clusters):
                column_names += [f"post_a{i}", f"lh_a{i}", f"prior_a{i}"]

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        self.file.write("\t".join(column_names) + "\n")

    def _write_sample(self, sample: Sample):
        feature_names = self.data.feature_names["external"]
        state_names = self.data.state_names["external"]

        if self.match_clusters:
            # Compute the best matching permutation
            permutation = get_best_permutation(sample.zones, self.cluster_sum)

            # Permute parameters
            p_zones = sample.p_zones[permutation, :, :]
            zones = sample.zones[permutation, :]

            # Update cluster_sum for matching future samples
            self.cluster_sum += zones
        else:
            # Unpermuted parameters
            p_zones = sample.p_zones
            zones = sample.zones

        row = {
            "Sample": sample.i_step,
            "posterior": sample.last_lh + sample.last_prior,
            "likelihood": sample.last_lh,
            "prior": sample.last_prior,
        }

        # Cluster sizes
        for i, cluster in enumerate(zones):
            col_name = f"size_a{i}"
            row[col_name] = np.count_nonzero(cluster)

        # weights
        for i_feat, feat_name in enumerate(feature_names):
            w_universal_name = f"w_universal_{feat_name}"
            w_contact_name = f"w_contact_{feat_name}"
            w_inheritance_name = f"w_inheritance_{feat_name}"

            row[w_universal_name] = sample.weights[i_feat, 0]
            row[w_contact_name] = sample.weights[i_feat, 1]
            if sample.inheritance:
                row[w_inheritance_name] = sample.weights[i_feat, 2]

        # alpha
        for i_feat, feat_name in enumerate(feature_names):
            for i_state, state_name in enumerate(state_names[i_feat]):
                col_name = f"alpha_{feat_name}_{state_name}"
                row[col_name] = sample.p_global[0, i_feat, i_state]

        # gamma
        for a in range(sample.n_clusters):
            for i_feat, feat_name in enumerate(feature_names):
                for i_state, state_name in enumerate(state_names[i_feat]):
                    col_name = f"gamma_a{(a + 1)}_{feat_name}_{state_name}"
                    row[col_name] = p_zones[a][i_feat][i_state]

        # beta
        if sample.inheritance:
            family_names = self.data.family_names["external"]
            for i_fam, fam_name in enumerate(family_names):
                for i_feat, feat_name in enumerate(feature_names):
                    for i_state, state_name in enumerate(state_names[i_feat]):
                        col_name = f"beta_{fam_name}_{feat_name}_{state_name}"
                        row[col_name] = sample.p_families[i_fam][i_feat][i_state]

        # lh, prior, posteriors
        if self.log_contribution_per_cluster:
            sample_single_cluster: Sample = sample.copy()

            for i in range(sample.n_clusters):
                sample_single_cluster.zones = zones[[i]]
                sample_single_cluster.everything_changed()
                lh = self.model.likelihood(sample_single_cluster, caching=False)
                prior = self.model.prior(sample_single_cluster)
                row[f"lh_a{i}"] = lh
                row[f"prior_a{i}"] = prior
                row[f"post_a{i}"] = lh + prior

        row_str = "\t".join([self.float_format % row[k] for k in self.column_names])
        self.file.write(row_str + "\n")


class ClustersLogger(ResultsLogger):

    """The ClustersLogger encodes each cluster in a bit-string and continually writes multiple
    clusters to a tab-separated text file."""

    def __init__(
        self,
        *args,
        match_clusters: bool = True,
    ):
        super().__init__(*args)
        self.match_clusters = match_clusters
        self.cluster_sum: Optional[npt.NDArray[int]] = None

    def write_header(self, sample: Sample):
        if sample.n_clusters <= 1:
            # Nothing to match
            self.match_clusters = False

        self.cluster_sum = np.zeros((sample.n_clusters, sample.n_sites), dtype=np.int)

    def _write_sample(self, sample):
        if self.match_clusters:
            # Compute best matching perm
            permutation = get_best_permutation(sample.zones, self.cluster_sum)

            # Permute zones
            zones = sample.zones[permutation, :]

            # Update cluster_sum for matching future samples
            self.cluster_sum += zones
        else:
            zones = sample.zones

        row = format_cluster_columns(zones)
        self.file.write(row + "\n")


class LikelihoodLogger(ResultsLogger):

    """The LikelihoodLogger continually writes the likelihood of each observation (one per
     site and feature) as a flattened array to a pytables file (.h5)."""

    def __init__(self, *args, **kwargs):
        self.logged_likelihood_array = None
        super().__init__(*args, **kwargs)

    def open(self):
        self.file = tables.open_file(self.path, mode="w")

    def write_header(self, sample: Sample):
        # Create the likelihood array
        self.logged_likelihood_array = self.file.create_earray(
            where=self.file.root,
            name="likelihood",
            atom=tables.Float32Col(),
            filters=tables.Filters(
                complevel=9, complib="blosc:zlib", bitshuffle=True, fletcher32=True
            ),
            shape=(0, sample.n_sites * sample.n_features),
        )

    def _write_sample(self, sample: Sample):
        self.logged_likelihood_array.append(sample.observation_lhs[None, ...])
