import os
import multiprocessing
from pathlib import Path

from sbayes.cli import main
from sbayes.tools.align_clusters import align_clusters
from sbayes.util import set_experiment_name

data_path = Path("./data")
parameters_path = Path("./parameters")
results_path = Path("./results")
sim_run_names = os.listdir(data_path)
sim_run_names.sort()
experiment_name = 'simu_run'

feature_types = "meta/feature_types.yaml"

run_args = []
for run in sorted(sim_run_names):
    features = data_path / run / "features_sim.csv"
    run_args.append(
        dict(
            config='config.yaml',
            experiment_name=run,
            custom_settings={
                'data': {
                    'features': features,
                    'feature_types': feature_types
                },
                'results': {
                    'path': results_path
                }
            }
        )
    )


def runner(kwargs):
    main(**kwargs)


# Run across multiple processes
pool = multiprocessing.Pool(processes=5)
pool.map(runner, run_args)


# Single-threaded version (for debugging):
for run_arg in run_args:
    runner(run_arg)


# Align inferred clusters with the simulated ones
for run in sorted(sim_run_names):
    clusters_sim = parameters_path / run / "clusters_sim.txt"
    # results_path.iterdir()
    K_dirs = [path for path in (results_path / run).iterdir() if os.path.isdir(path)]
    assert len(K_dirs) == 1, K_dirs
    K_dir = K_dirs[0]
    K = K_dir.name
    clusters_inferred = K_dir / f"clusters_{K}_0.txt"
    parameters_inferred = K_dir / f"stats_{K}_0.txt"
    align_clusters(
        clusters_sim,
        clusters_inferred,
        parameters_inferred,
        clusters_inferred,
        parameters_inferred,
    )
