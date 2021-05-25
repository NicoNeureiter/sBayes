import numpy as np

from sbayes.sampling.zone_sampling import Sample
from util import format_area_columns


class ResultsLogger(object):

    def __init__(self, path, data):
        self.path = path
        self.file = open(path, 'w')
        self.data = data
        self.column_names = None

    def write_header(self, sample):
        raise NotImplementedError

    def write_sample(self, sample):
        raise NotImplementedError

    def close(self):
        self.file.close()


class ParametersLogger(ResultsLogger):

    def write_header(self, sample: Sample):
        feature_names = self.data.feature_names['external']
        state_names = self.data.state_names['external']
        family_names = self.data.family_names['external']

        column_names = ['Sample', 'posterior', 'likelihood', 'prior']

        # Area sizes
        for i in range(sample.n_areas):
            column_names.append(f'size_a{i}')

        # weights
        for i_feat, feat_name in enumerate(feature_names):
            column_names.append(f'w_universal_{feat_name}')
            column_names.append(f'w_contact_{feat_name}')
            if sample.inheritance:
                column_names.append(f'w_inheritance_{feat_name}')

        # alpha
        for i_feat, feat_name in enumerate(feature_names):
            for i_state, state_name in enumerate(state_names[i_feat]):
                col_name = f'alpha_{feat_name}_{state_name}'
                column_names.append(col_name)

        # gamma
        for a in range(sample.n_areas):
            for i_feat, feat_name in enumerate(feature_names):
                for i_state, state_name in enumerate(state_names[i_feat]):
                    col_name = f'gamma_a{(a + 1)}_{feat_name}_{state_name}'
                    column_names.append(col_name)

        # beta
        if sample.inheritance:
            for i_fam, fam_name in enumerate(family_names):
                for i_feat, feat_name in enumerate(feature_names):
                    for i_state, state_name in enumerate(state_names[i_feat]):
                        col_name = f'beta_{fam_name}_{feat_name}_{state_name}'
                        column_names += [col_name]

        # Store the column names in an attribute (important to keep order consistent)
        self.column_names = column_names

        # Write the column names to the logger file
        self.file.write('\t'.join(column_names) + '\n')

    def write_sample(self, sample):
        feature_names = self.data.feature_names['external']
        state_names = self.data.state_names['external']
        family_names = self.data.family_names['external']

        row = {
            'Sample': sample.i_sample,
            'posterior': sample.likelihood + sample.prior,  # (everything in log-space)
            'likelihood': sample.likelihood,
            'prior': sample.prior,
        }

        # Area sizes
        for i, area in enumerate(sample.zones):
            col_name = f'size_a{i}'
            row[col_name] = np.count_nonzero(area)

        # weights
        for i_feat, feat_name in enumerate(feature_names):
            w_universal_name = f'w_universal_{feat_name}'
            w_contact_name = f'w_contact_{feat_name}'
            w_inheritance_name = f'w_inheritance_{feat_name}'

            row[w_universal_name] = sample.weights[i_feat, 0]
            row[w_contact_name] = sample.weights[i_feat, 1]
            if sample.inheritance:
                row[w_inheritance_name] = sample.weights[i_feat, 2]

        # alpha
        for i_feat, feat_name in enumerate(feature_names):
            for i_state, state_name in enumerate(state_names[i_feat]):
                col_name = f'alpha_{feat_name}_{state_name}'
                row[col_name] = sample.p_global[0, i_feat, i_state]

        # gamma
        for a in range(sample.n_areas):
            for i_feat, feat_name in enumerate(feature_names):
                for i_state, state_name in enumerate(state_names[i_feat]):
                    col_name = f'gamma_a{(a + 1)}_{feat_name}_{state_name}'
                    row[col_name] = sample.p_zones[a][i_feat][i_state]

        # beta
        if sample.inheritance:
            for i_fam, fam_name in enumerate(family_names):
                for i_feat, feat_name in enumerate(feature_names):
                    for i_state, state_name in enumerate(state_names[i_feat]):
                        col_name = f'beta_{fam_name}_{feat_name}_{state_name}'
                        row[col_name] = sample.p_families[i_fam][i_feat][i_state]

        row_str = '\t'.join([str(row[k]) for k in self.column_names])
        self.file.write(row_str + '\n')

        # TODO Do this in post-processing:
        # # Recall and precision
        # if data.is_simulated:
        #     sample_z = np.any(samples['sample_zones'][s], axis=0)
        #     true_z = np.any(samples['true_zones'], axis=0)
        #     n_true = np.sum(true_z)
        #     intersections = np.minimum(sample_z, true_z)
        #
        #     total_recall = np.sum(intersections, axis=0) / n_true
        #     precision = np.sum(intersections, axis=0) / np.sum(sample_z, axis=0)
        #
        #     column_names += ['recall']
        #     row['recall'] = total_recall
        #
        #     column_names += ['precision']
        #     row['precision'] = precision
        #
        # # Single areas
        # if 'sample_lh_single_zones' in samples.keys():
        #     for a in range(sample.n_areas):
        #         lh_name = 'lh_a' + str(a + 1)
        #         prior_name = 'prior_a' + str(a + 1)
        #         posterior_name = 'post_a' + str(a + 1)
        #
        #         column_names += [lh_name]
        #         row[lh_name] = samples['sample_lh_single_zones'][s][a]
        #
        #         column_names += [prior_name]
        #         row[prior_name] = samples['sample_prior_single_zones'][s][a]
        #
        #         column_names += [posterior_name]
        #         row[posterior_name] = samples['sample_posterior_single_zones'][s][a]


class AreasLogger(ResultsLogger):

    def write_header(self, sample):
        pass

    def write_sample(self, sample):
        row = format_area_columns(sample.zones)
        self.file.write(row + '\n')

