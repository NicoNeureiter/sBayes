import numpy as np
import pandas as pd

from jax import Array, random
from numpyro.infer import Predictive
from sbayes.load_data import Experiment, Data, write_feature_types
from sbayes.mcmc_setup import MCMCSetup
from sbayes.model import Model
from sbayes.config.config import parameter_transformations
from sbayes.simulate.io import (prepare_output_folders, write_simulated_data,
                                write_simulated_parameters)
from string import ascii_uppercase


from sbayes.simulate.config import SimulateConfigFile, get_feature_types, feature_types_by_name


class Simulator:
    """Simulate data sets from an sBayes model and optionally run inference on them.

    Draws parameters and features from the prior of the model defined by the sBayes
    config, so the ground truth is known. Use in this order: `prepare_simulation`,
    `simulate`, `write_simulation`, and optionally `infer`.
    """

    def __init__(self, config: SimulateConfigFile):
        """
        Args:
            config: the simulation config, with the sBayes config loaded into
                `inf_config` (see `load_config`)
        """
        if config.inf_config is None:
            raise ValueError(
                "The sBayes config has not been loaded. Use `load_config` to read the "
                "simulation config."
            )

        self.config = config
        self.data: Data | None = None
        self.model: Model | None = None
        self.prior_samples: dict[str, np.ndarray] | None = None
        self.feature_types: dict[str, dict] | None = None
        self._empty_csv: pd.DataFrame | None = None


    def prepare_simulation(self) -> None:
        """Build the model to simulate from.

        Creates the object, confounder and feature structure of the data to simulate,
        then builds the sBayes model on it in simulation mode: `sample_from_prior` so
        that the parameters come from their priors, and `simulate_observations` so that
        the observation sites stay latent and `Predictive` draws the features.
        """

        sim_config = self.config.inf_config.model_copy(deep=True)
        sim_config.model.sample_from_prior = True
        sim_config.model.clusters = self.config.simulation.clusters

        rng = np.random.default_rng(self.config.simulation.seed)
        empty_csv = self.create_feature_structure(rng)
        feature_types = get_feature_types(self.config.simulation.features)
        self._empty_csv = empty_csv
        self.feature_types = feature_types

        data = Data.from_simulation(
            features_csv=empty_csv,
            feature_types=feature_types,
            config=sim_config,
            logger=None,
        )
        self.data = data
        self.model = Model(data=data, config=sim_config.model, simulate_observations=True)


    def create_feature_structure(self, rng: np.random.Generator) -> pd.DataFrame:
        """Build the data frame that the simulated features are written into.

        Objects get an id, a name and random coordinates. Confounder groups are
        assigned in a fixed rotation, so that the groups are balanced to within one
        object and identical across replicates. The feature columns are created empty;
        they are filled with values drawn from the prior in `write_simulation`.

        Args:
            rng: the random generator, seeded from the simulation config. Used for the
                coordinates only; the group assignment is deterministic.

        Returns:
            One row per object, with the metadata, confounder and empty feature columns.

        Raises:
            ValueError: if a confounder has fewer objects than groups, so that a group
                would necessarily stay empty.
        """
        simulation = self.config.simulation
        n_objects = simulation.objects

        columns: dict[str, object] = {
            "id": list(range(n_objects)),
            "name": [f"object_{i}" for i in range(n_objects)],
            "x": rng.uniform(simulation.x_min, simulation.x_max, size=n_objects),
            "y": rng.uniform(simulation.y_min, simulation.y_max, size=n_objects),
        }

        for confounder, n_groups in simulation.confounders.items():
            if n_groups > n_objects:
                raise ValueError(
                    f"Confounder '{confounder}' has {n_groups} groups but only "
                    f"{n_objects} objects are simulated."
                )
            if n_groups > len(ascii_uppercase):
                raise ValueError(
                    f"Confounder '{confounder}' has {n_groups} groups, but group "
                    f"labels are single letters (at most {len(ascii_uppercase)})."
                )
            # A fixed rotation keeps the groups balanced and reproducible. Random
            # assignment can leave a group small by chance, which weakens the
            # confounder's identifiability for no reason.
            columns[confounder] = [
                f"group_{ascii_uppercase[i % n_groups]}" for i in range(n_objects)
            ]

        for feature in feature_types_by_name(simulation.features):
            columns[feature] = [None] * n_objects

        return pd.DataFrame(columns)


    def simulate(self, rng_key: Array | None = None) -> None:
        """Draw data sets from the prior of the model.

        Draws `simulation.datasets` samples of both the parameters and the features,
        keeping them in memory for `write_simulation` to write out.

        Args:
            rng_key: the JAX random key. Defaults to a key built from the `seed` in the
                simulation config, which also seeds the locations and confounder groups.

        Raises:
            RuntimeError: if `prepare_simulation` has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call prepare_simulation() before simulate().")

        if rng_key is None:
            rng_key = random.PRNGKey(self.config.simulation.seed)

        predictive = Predictive(
            self.model.get_model, num_samples=self.config.simulation.datasets
        )
        self.prior_samples = predictive(rng_key)


    def write_simulation(
        self, write_data: bool = True, write_parameters: bool = True
    ) -> None:
        """Write the simulated data sets to the output directory.

        Each data set goes into its own `sim_{index}` folder, with the features in
        `sim_data` and the parameters in `sim_params`. The parameters are written in
        the same format as inferred samples, so the two can be compared directly. The
        feature types are written once, next to the data sets.

        Args:
            write_data: whether to write the simulated features
            write_parameters: whether to write the simulated parameters

        Raises:
            RuntimeError: if `simulate` has not been called.
            ValueError: if neither data nor parameters are to be written.
        """
        if self.prior_samples is None or self.data is None or self.model is None \
                or self._empty_csv is None or self.feature_types is None:
            raise RuntimeError("Call prepare_simulation() and simulate() before write_simulation().")

        if not (write_data or write_parameters):
            raise ValueError("At least one of write_data and write_parameters must be True.")

        prior_samples = self.prior_samples
        empty_csv = self._empty_csv
        simulation = self.config.simulation
        output_path = simulation.output.path

        output_path.mkdir(parents=True, exist_ok=True)

        if write_data:
            write_feature_types(self.feature_types, output_path / "feature_types.yaml")

        # The cluster assignments written by write_samples are drawn with NumPy's
        # global generator
        np.random.seed(simulation.seed)

        for s in range(simulation.datasets):
            sample_index = None if simulation.datasets == 1 else s
            folder_parameters, folder_data = prepare_output_folders(output_path, sample_index)

            # Keep the leading sample axis, which the writers read as one sample
            prior_sample = {k: v[s:s + 1] for k, v in prior_samples.items()}

            if write_parameters:
                write_simulated_parameters(
                    sample=prior_sample,
                    data=self.data,
                    model=self.model,
                    base_path=folder_parameters,
                )
            if write_data:
                write_simulated_data(
                    partitions=self.data.features.partitions,
                    sample=prior_sample,
                    features_csv=empty_csv.copy(deep=True),
                    base_path=folder_data,
                )


    def infer(self, sample_from_prior: bool = False) -> None:
        """Run inference on every simulated data set.

        Args:
            sample_from_prior: whether to sample from the prior for the no-data inference check

        Runs the sampler on each data set and cluster count in the sBayes config, so
        the inferred parameters can be compared against the simulated ground truth.
        The results of one data set go into its own folder, next to the data they were
        inferred from:

            <output>/sim_{s}/results/simK{simulated}/K{inferred}/

        A single data set has no `sim_{s}` level and goes directly into `<output>`.
        Raises:
            RuntimeError: if `write_simulation` has not written the data sets.
        """

        simulation = self.config.simulation
        inf_config = self.config.inf_config
        simulation_path = simulation.output.path


        clusters = inf_config.model.clusters
        if isinstance(clusters, int):
            clusters = [clusters]

        for s in range(simulation.datasets):
            sample_index = None if simulation.datasets == 1 else s
            _, data_folder = prepare_output_folders(simulation_path, sample_index)
            features_path = data_folder / "features.csv"
            if not features_path.exists():
                raise RuntimeError(
                    f"No simulated data at {features_path}. Call write_simulation() first."
                )

            # Keep the results of a data set next to the data they were inferred from
            results_path = features_path.parent.parent / "results"
            for k in clusters:
                experiment = Experiment(
                    config_file=self.config.config_path,
                    # The ground truth; the inferred k becomes the folder below this one
                    experiment_name=f"simK{simulation.clusters}",
                    custom_settings={
                        "results": {"path": str(results_path)},
                        "data": {
                            "features": str(features_path),
                            "feature_types": str(simulation_path / "feature_types.yaml"),
                        },
                        "model": {"clusters": k, "sample_from_prior": sample_from_prior},
                        "mcmc": {"runs": 1},  # one chain per run; the loop below runs them
                    },
                    log=True,
                    copy_config=False,
                )
                if sample_from_prior:

                    flags = parameter_transformations(experiment.config)
                    if flags != {False}:
                        raise ValueError(
                            f"Prior sampling requires the untransformed "
                            f"parameterization, but the priors give {flags}."
                        )

                data = Data.from_experiment(experiment)
                mcmc = MCMCSetup(data, experiment)
                for i_run in range(inf_config.mcmc.runs):
                    mcmc.sample(run=i_run, resume=False)
