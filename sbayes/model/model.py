#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from sbayes.model.model_shapes import ModelShapes
from sbayes.model.prior import Prior
from sbayes.model.likelihood import Likelihood
from sbayes.config.config import ModelConfig
from sbayes.load_data import Data


class Model:
    """The sBayes model: posterior distribution of clusters and parameters.

    Attributes:
        data (Data): The data used in the likelihood
        config (ModelConfig): A dictionary containing configuration parameters of the model
        confounders (dict): A dict of all confounders and group names
        shapes (ModelShapes): A dictionary with shape information for building the Likelihood and Prior objects
        likelihood (Likelihood): The likelihood of the model
        prior (Prior): The prior of the model

    """
    def __init__(self, data: Data, config: ModelConfig):
        self.data = data
        self.config = config
        self.confounders = data.confounders
        self.n_clusters = config.clusters
        self.min_size = config.prior.objects_per_cluster.min
        self.max_size = config.prior.objects_per_cluster.max
        self.sample_source = config.sample_source

        try:
            n_categorical = data.features.categorical.n_features
            n_states = data.features.categorical.states.shape[1]
            states_per_feature = data.features.categorical.states
        except AttributeError:
            n_categorical = None
            n_states = None
            states_per_feature = None

        try:
            n_gaussian = data.features.gaussian.n_features
        except AttributeError:
            n_gaussian = None

        try:
            n_poisson = data.features.poisson.n_features
        except AttributeError:
            n_poisson = None

        try:
            n_logit_normal = data.features.logitnormal.n_features
        except AttributeError:
            n_logit_normal = None

        self.shapes = ModelShapes(
            n_objects=self.data.objects.n_objects,
            n_clusters=self.n_clusters,
            n_confounders=len(self.confounders),
            n_groups={name: conf.n_groups for name, conf in self.confounders.items()},
            n_features=data.features.n_features,
            n_features_categorical=n_categorical,
            n_features_gaussian=n_gaussian,
            n_features_poisson=n_poisson,
            n_features_logitnormal=n_logit_normal,
            n_states_categorical=n_states,
            states_per_feature=states_per_feature
        )

        # Create likelihood and prior objects
        self.prior = Prior(data=data, config=self.config.prior, shapes=self.shapes, sample_source=self.sample_source)
        self.likelihood = Likelihood(data=self.data, shapes=self.shapes, prior=self.prior)

    def __call__(self, sample, caching=True):
        """Evaluate the (non-normalized) posterior probability of the given sample."""

        log_likelihood = self.likelihood(sample, caching=caching)
        log_prior = self.prior(sample)
        return log_likelihood + log_prior

    def __copy__(self):
        return Model(self.data, self.config)

    def get_setup_message(self):
        """Compile a set-up message for logging."""
        setup_msg = "\n"
        setup_msg += "Model\n"
        setup_msg += "##########################################\n"
        setup_msg += f"Number of clusters: {self.config.clusters}\n"
        setup_msg += f"Clusters have a minimum size of {self.config.prior.objects_per_cluster.min} " \
                     f"and a maximum size of {self.config.prior.objects_per_cluster.max}\n"
        setup_msg += self.prior.get_setup_message()
        return setup_msg
