# sBayes

This software package implements a Bayesian mixture model for reconstructing linguistic contact areas, as 
described in 
[Contact-tracing in cultural evolution: a Bayesian mixture model to detect geographic areas of language contact](https://www.biorxiv.org/content/10.1101/2021.03.31.437731v3)
(Ranacher P., Neureiter N., Van Gijn R., Sonnenhauser B., Escher A., Weibel R., Muysken P., Bickel B.).
*sBayes* implements a custom MCMC sampler to generate contact areas according to the model. Here we describe
the installation process and the basic commands needed to run an analysis. For more detailed instructions explaining
each step in the analysis and the various settings, please consult the [user manual](documentation/user_manual.md).


## Installation
To run sBayes, you need Python (version >=3.7) and three required system libraries: `GEOS`, `GDAL`, `PROJ`. The way of
installing these system requirements depends on your operating system. E.g. on Linux (Ubuntu/Debian) you can use the 
following command:
```shell
sudo apt-get install -y libproj-dev proj-data proj-bin libgeos-dev
```

On MAC OS the same can be done using the homebrew package manager:
```shell
brew install proj geos gdal
```

On Windows the three libraries need to be installed manually from the corresponding sources.

Once these system libraries are ready, sBayes can be installed (along with the required python libraries) using:
```shell
 pip install --user git+git://github.com/derpetermann/sBayes.git
```

## Running sBayes
sBayes can be used as a python library or through a command line interface. Here we 
describe the command line interface, which offers a convenient way to run the standard 
workflow of a sBayes analysis. To run sBayes from the command line, simply call: 
```shell
sbayes <path_to_config_file>
```
The config file is a JSON file which contains all the settings for the analysis. The 
results are written to CSV files, which can be visualized using the plotting functions.
The format of the config file and details about the set-up and visualization are described in
the user manual.
