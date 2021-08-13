#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines the class ContactAreasSimulator
    Outputs a simulated contact areas, together with network, features,
    states, families, weights, p_universal (alpha), p_inheritance(beta), p_contact(gamma) """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from sbayes.preprocessing import (compute_network, read_sites,
                                  simulate_assignment_probabilities,
                                  assign_family,
                                  assign_area,
                                  simulate_features,
                                  simulate_weights,
                                  subset_features,
                                  counts_from_complement)
from sbayes.util import assess_correlation_probabilities


class Simulation:
    def __init__(self, experiment, n_correlated=10):

        self.path_log = experiment.path_results / 'experiment.log'
        self.config = experiment.config['simulation']

        self.sites_file = experiment.config['simulation']['sites']
        self.log_read_sites = None

        # Simulated parameters
        self.sites = None
        self.network = None
        self.areas = None
        self.features = None
        self.states = None
        self.families = None
        self.weights = None
        self.p_universal = None
        self.p_contact = None
        self.p_inheritance = None
        self.inheritance = None
        self.subset = None
        self.prior_universal = None
        self.geo_prior = None

        self.feature_names = None
        self.state_names = None
        self.family_names = None
        self.site_names = None

        # Is a simulation
        self.is_simulated = True

        # Correlation between features
        self.corr_th = experiment.config['simulation']['correlation_threshold']
        self.n_correlated = n_correlated

    def log_simulation(self):
        logging.basicConfig(format='%(message)s', filename=self.path_log, level=logging.DEBUG)
        logging.info("\n")
        logging.info("SIMULATION")
        logging.info("##########################################")
        logging.info(self.log_read_sites)
        logging.info("Inheritance is simulated: %s", self.config['inheritance'])
        logging.info("Simulated features: %s", self.config['n_features'])
        logging.info("Simulated intensity for universal pressure: %s", self.config['i_universal'])
        logging.info("Simulated intensity for contact: %s", self.config['i_contact'])
        logging.info("Simulated intensity for inheritance: %s", self.config['i_inheritance'])
        logging.info("Simulated level of entropy for universal pressure: %s", self.config['e_universal'])
        logging.info("Simulated level of entropy for contact: %s", self.config['e_contact'])
        logging.info("Simulated level of entropy for inheritance: %s", self.config['e_inheritance'])
        logging.info("Simulated area: %s", self.config['area'])

    def run_simulation(self):

        self.inheritance = self.config['inheritance']
        self.subset = self.config['subset']

        # Get sites
        self.sites, self.site_names, self.log_read_sites = read_sites(file=self.sites_file,
                                                                      retrieve_family=self.inheritance,
                                                                      retrieve_subset=self.subset)

        self.network = compute_network(self.sites)
        self.areas = assign_area(area_id=self.config['area'], sites_sim=self.sites)

        # Simulate families
        if self.inheritance:
            self.families, self.family_names = assign_family(fam_id=1, sites_sim=self.sites)
        else:
            self.families = None

        # Simulate weights, i.e. the influence of universal pressure, contact and inheritance on each feature
        self.weights = simulate_weights(i_universal=self.config['i_universal'],
                                        i_contact=self.config['i_contact'],
                                        i_inheritance=self.config['i_inheritance'],
                                        inheritance=self.inheritance,
                                        n_features=self.config['n_features'])
        attempts = 0
        while True:
            attempts += 1

            # Simulate probabilities for features to be universally preferred,
            # passed through contact (and inherited if available)
            self.p_universal, self.p_contact, self.p_inheritance \
                = simulate_assignment_probabilities(e_universal=self.config['e_universal'],
                                                    e_contact=self.config['e_contact'],
                                                    e_inheritance=self.config['e_inheritance'],
                                                    inheritance=self.inheritance,
                                                    n_features=self.config['n_features'],
                                                    p_number_categories=self.config['p_n_categories'],
                                                    areas=self.areas, families=self.families)

            correlated = assess_correlation_probabilities(self.p_universal, self.p_contact, self.p_inheritance,
                                                          corr_th=self.corr_th)

            if correlated <= self.n_correlated:
                break

            if attempts > 10000:
                attempts = 0

                self.corr_th += 0.05
                self.n_correlated += 1
                print("Correlation threshold for simulation increased to", self.corr_th)
                print("Number of allowed correlated features increased to", self.n_correlated)

        # Simulate features
        self.features, self.states, self.feature_names, self.state_names = \
            simulate_features(areas=self.areas,
                              families=self.families,
                              p_universal=self.p_universal,
                              p_contact=self.p_contact,
                              p_inheritance=self.p_inheritance,
                              weights=self.weights,
                              inheritance=self.inheritance)

        if self.subset:
            # The data is split into two parts: subset and complement
            # The subset is used for analysis and the complement to define the prior
            counts = counts_from_complement(features=self.features,
                                            subset=self.sites['subset'])

            self.prior_universal = {'counts': counts,
                                    'states': self.states}

            self.network = compute_network(sites=self.sites, subset=self.sites['subset'])
            sub_idx = np.nonzero(self.sites['subset'])[0]
            self.areas = self.areas[np.newaxis, 0, sub_idx]
            self.features = subset_features(features=self.features, subset=self.sites['subset'])

        geo_cost_matrix = self.network['dist_mat']
        self.geo_prior = {'cost_matrix': geo_cost_matrix}
