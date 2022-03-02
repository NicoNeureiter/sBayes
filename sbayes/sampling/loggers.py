from typing import TextIO, Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import tables

from sbayes.load_data import Data
from sbayes.util import format_area_columns, get_best_permutation
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
        self.file = open(self.path, "w")

    def close(self):
        self.file.close()
        self.file = None


class ParametersCSVLogger(ResultsLogger):

    """The ParametersCSVLogger collects all real-valued parameters (weights, alpha, beta,
    gamma) and some statistics (area size, likelihood, prior, posterior) and continually
    writes them to a tab-separated text-file."""

    def __init__(
        self,
        *args,
        log_contribution_per_area: bool = True,
        float_format: str = "%.10g",
        match_areas: bool = True,
    ):
        super().__init__(*args)
        self.float_format = float_format
        self.log_contribution_per_area = log_contribution_per_area
        self.match_areas = match_areas
        self.area_sum: Optional[npt.NDArray[int]] = None

    def write_header(self, sample):
        feature_names = self.data.feature_names["external"]
        state_names = self.data.state_names["external"]
        column_names = ["Sample", "posterior", "likelihood", "prior"]

        # No need for matching if only 1 area (or no areas at all)
        if sample.n_areas <= 1:
            self.match_areas = False

        # Initialize area_sum array for matching
        self.area_sum = np.zeros((sample.n_areas, sample.n_sites), dtype=np.int)

        # Area sizes
        for i in range(sample.n_areas):
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
        for a in range(sample.n_areas):
            for i_feat, feat_name in enumerate(feature_names):
                for i_state, state_name in enumerate(state_names[i_feat]):
                    col_name = f"gamma_a{(a + 1)}_{feat_name}_{state_name}"
                    column_names.append(col_name)

        # beta
        if sample.inheritance:
            family_names = self.data.family_names["external"]
            for i_fam, fam_name in enumerate(family_names):
                for i_feat, feat_name in enumerate(feature_names):
                    for i_state, state_name in enumerate(state_names[i_feat]):
                        col_name = f"beta_{fam_name}_{feat_name}_{state_name}"
                        column_names += [col_name]

        # lh, prior, posteriors
        if self.log_contribution_per_area:
            for i in range(sample.n_areas):
                column_names += [f"post_a{i}", f"lh_a{i}", f"prior_a{i}"]

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        self.file.write("\t".join(column_names) + "\n")

    def _write_sample(self, sample: Sample):
        feature_names = self.data.feature_names["external"]
        state_names = self.data.state_names["external"]

        if self.match_areas:
            # Compute the best matching permutation
            permutation = get_best_permutation(sample.zones, self.area_sum)

            # Permute parameters
            p_zones = sample.p_zones[permutation, :, :]
            zones = sample.zones[permutation, :]

            # Update area_sum for matching future samples
            self.area_sum += zones
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

        # Area sizes
        for i, area in enumerate(zones):
            col_name = f"size_a{i}"
            row[col_name] = np.count_nonzero(area)

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
        for a in range(sample.n_areas):
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
        if self.log_contribution_per_area:
            sample_single_area: Sample = sample.copy()

            for i in range(sample.n_areas):
                sample_single_area.zones = zones[[i]]
                sample_single_area.everything_changed()
                lh = self.model.likelihood(sample_single_area, caching=False)
                prior = self.model.prior(sample_single_area)
                row[f"lh_a{i}"] = lh
                row[f"prior_a{i}"] = prior
                row[f"post_a{i}"] = lh + prior

        row_str = "\t".join([self.float_format % row[k] for k in self.column_names])
        self.file.write(row_str + "\n")


class AreasLogger(ResultsLogger):

    """The AreasLogger encodes each area in a bit-string and continually writes multiple
    areas to a tab-separated text file."""

    def __init__(
        self,
        *args,
        match_areas: bool = True,
    ):
        super().__init__(*args)
        self.match_areas = match_areas
        self.area_sum: Optional[npt.NDArray[int]] = None

    def write_header(self, sample: Sample):
        if sample.n_areas <= 1:
            # Nothing to match
            self.match_areas = False

        self.area_sum = np.zeros((sample.n_areas, sample.n_sites), dtype=np.int)

    def _write_sample(self, sample):
        if self.match_areas:
            # Compute best matching perm
            permutation = get_best_permutation(sample.zones, self.area_sum)

            # Permute zones
            zones = sample.zones[permutation, :]

            # Update area_sum for matching future samples
            self.area_sum += zones
        else:
            zones = sample.zones

        row = format_area_columns(zones)
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
