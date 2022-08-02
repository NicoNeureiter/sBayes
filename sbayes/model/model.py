#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass

from numpy.typing import NDArray

from sbayes.model.prior import Prior
from sbayes.model.likelihood import Likelihood
from sbayes.config.config import ModelConfig
from sbayes.load_data import Data


@dataclass
class ModelShapes:
    n_clusters: int
    n_sites: int
    n_features: int
    n_states: int
    states_per_feature: NDArray[bool]

    @property
    def n_states_per_feature(self):
        return [sum(applicable) for applicable in self.states_per_feature]

    def __getitem__(self, key):
        """Getter for backwards compatibility with dict-notation."""
        return getattr(self, key)


class Model:
    """The sBayes model: posterior distribution of clusters and parameters.

    Attributes:
        data (Data): The data used in the likelihood
        config (ModelConfig): A dictionary containing configuration parameters of the model
        confounders (dict): A ict of all confounders and group names
        shapes (ModelShapes): A dictionary with shape information for building the Likelihood and Prior objects
        likelihood (Likelihood): The likelihood of the model
        prior (Prior): Rhe prior of the model

    """
    def __init__(self, data: Data, config: ModelConfig):
        self.data = data
        self.config = config
        self.confounders = config.confounders
        self.n_clusters = config.clusters
        self.min_size = config.prior.objects_per_cluster.min
        self.max_size = config.prior.objects_per_cluster.max
        self.sample_source = config.sample_source
        n_sites, n_features, n_states = self.data.features.values.shape

        self.shapes = ModelShapes(
            n_clusters=self.n_clusters,
            n_sites=n_sites,
            n_features=n_features,
            n_states=n_states,
            states_per_feature=self.data.features.states
        )

        # Create likelihood and prior objects
        self.likelihood = Likelihood(data=self.data, shapes=self.shapes)
        self.prior = Prior(shapes=self.shapes, config=self.config.prior, data=data, sample_source=self.sample_source)

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
