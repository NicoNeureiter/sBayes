from typing import TextIO, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np
import tables

from sbayes.load_data import Data
from sbayes.util import format_area_columns

Sample = "sbayes.sampling.zone_sampling.Sample"


class ResultsLogger(ABC):
    def __init__(
        self,
        path: str,
        data: Data,
    ):
        self.path: str = path
        self.data: Data = data

        self.file: Optional[TextIO] = None
        self.column_names: Optional[list] = None

    def write_header(self, sample: Sample):
        raise NotImplementedError

    @abstractmethod
    def _write_sample(self, sample: Sample):
        pass

    def write_sample(self, sample: Sample):
        if self.file is None:
            self.initialize(sample)
        self._write_sample(sample)

    def initialize(self, sample: Sample):
        self.open()
        self.write_header(sample)

    def open(self):
        self.file = open(self.path, "w")

    def close(self):
        self.file.close()
        self.file = None


class ParametersCSVLogger(ResultsLogger):
    def __init__(
        self,
        path: str,
        data: Data,
        float_format: str = "%.12g",
    ):
        super(ParametersCSVLogger, self).__init__(path=path, data=data)
        self.float_format = float_format

    def write_header(self, sample):
        feature_names = self.data.feature_names["external"]
        state_names = self.data.state_names["external"]
        column_names = ["Sample", "posterior", "likelihood", "prior"]

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

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        self.file.write("\t".join(column_names) + "\n")

    def _write_sample(self, sample: Sample):
        feature_names = self.data.feature_names["external"]
        state_names = self.data.state_names["external"]

        row = {
            "Sample": sample.i_step,
            "posterior": sample.last_lh + sample.last_prior,
            "likelihood": sample.last_lh,
            "prior": sample.last_prior,
        }

        # Area sizes
        for i, area in enumerate(sample.zones):
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
                    row[col_name] = sample.p_zones[a][i_feat][i_state]

        # beta
        if sample.inheritance:
            family_names = self.data.family_names["external"]
            for i_fam, fam_name in enumerate(family_names):
                for i_feat, feat_name in enumerate(feature_names):
                    for i_state, state_name in enumerate(state_names[i_feat]):
                        col_name = f"beta_{fam_name}_{feat_name}_{state_name}"
                        row[col_name] = sample.p_families[i_fam][i_feat][i_state]

        row_str = "\t".join([self.float_format % row[k] for k in self.column_names])
        self.file.write(row_str + "\n")


class AreasLogger(ResultsLogger):
    def write_header(self, sample: Sample):
        pass

    def _write_sample(self, sample):
        row = format_area_columns(sample.zones)
        self.file.write(row + "\n")


class LikelihoodLogger(ResultsLogger):
    def __init__(self, *args, **kwargs):
        self.logged_likelihood_array = None
        super(LikelihoodLogger, self).__init__(*args, **kwargs)

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

    def open(self):
        self.file = tables.open_file(self.path, mode="w")
