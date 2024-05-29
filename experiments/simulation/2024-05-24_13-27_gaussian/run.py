import os
from sbayes.cli import main
from sbayes.util import set_experiment_name

data_path = "../2024-05-24_13-27_gaussian/data"
res_path = "../2024-05-24_13-27_gaussian/results"
n_sim = os.listdir(data_path)
experiment_name = set_experiment_name()

feature_types = "feature_types.yaml"

for i in n_sim:

    features = os.path.join(data_path, i, "features_sim.csv")

    result_path = os.path.join(res_path, i)
    main(config='config.yaml',
         experiment_name=experiment_name,
         custom_settings=dict(
             data=dict(
                 features=features,
                 feature_types=feature_types),
             results=dict(
                 path=result_path)))
