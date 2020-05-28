### *class* **DataImporter**

Create an object which imports the features and the prior probabilities.
The file containing features should be named ```features.csv``` and should be stored 
in the folder ```../../<region_full>/data/features```. 
The variable ```<region_full>``` is to be specified in the ```config_data.json``` (as well as ```<region_abbr>``` for the logging file).

Other files needed for the data import are prior probabilities *.csv files which should be stored in the following folders:
```
../../<region_full>/data/p_families
../../<region_full>/data/p_global
```
The variables assigned in the class methods are used further in the MCMC process for identifying the contact zones.

The class **DataImporter** defines the following functions:

*DataImporter*.**logging_setup()**

Log the setup information, which includes the name of the experiment (date and time by default), the value of the parameter INHERITANCE and the information on the imported data.
The log file ```info.log``` is stored in the folder ```../results/contact_zones/<experiment_name>```.
This method should be called before the methods *DataImporter.get_data_features()* and *DataImporter.get_prior_information()* in order to prepare the log file.

*DataImporter*.**get_parameters()**

Read the file ```config_data.json``` and store the configuration information in the dictionary *DataImporter*.config. This function is called within the *\_\_init\_\_* function of the class.

*DataImporter*.**get_data_features()**

Read the features from the file: 
```../../<region_full>/data/features/features.csv```

Assign the variables:
- *DataImporter*.sites
- *DataImporter*.site_names
- *DataImporter*.features
- *DataImporter*.feature_names
- *DataImporter*.category_names
- *DataImporter*.families
- *DataImporter*.family_names
- *DataImporter*.network

*DataImporter*.**get_prior_information()**

Read the prior information from the *.csv files stored in the folders:
```
../../<region_full>/data/p_families
../../<region_full>/data/p_global
```

Assign the variables:
- *DataImporter*.p_global_dirichlet
- *DataImporter*.p_global_categories
- *DataImporter*.p_families_dirichlet
- *DataImporter*.p_families_categories

A short usage example:

```
from import_data import DataImporter

di = DataImporter()
di.logging_setup()
di.get_data_features()
di.get_prior_information()
```

### *class* **ContactZonesSimulator**

Create an object which simulates the features and the prior probabilities.
The class uses the file ```sites_simulation.csv```, which should be stored in the folder ```../data```.
The variables' values simulated in the class methods are used further in the MCMC process for identifying the contact zones.

The class **ContactZonesSimulator** defines the following functions:

*ContactZonesSimulator*.**get_parameters()**

Read the file ```config_data.json``` and store the configuration information in the dictionary *ContactZonesSimulator*.config. This function is called within the *\_\_init\_\_* function of the class.

*ContactZonesSimulator*.**simulation()**

Read the file ```../data/sites_simulation.csv```, simulate the features and the prior probabilities. Assign the following variables:

- *ContactZonesSimulator*.network
- *ContactZonesSimulator*.zones
- *ContactZonesSimulator*.families
- *ContactZonesSimulator*.weights
- *ContactZonesSimulator*.p_global
- *ContactZonesSimulator*.p_zones
- *ContactZonesSimulator*.p_families
- *ContactZonesSimulator*.features
- *ContactZonesSimulator*.categories

*ContactZonesSimulator*.**logging_setup()**

Log the setup information, which includes the name of the experiment (date and time by default), the value of the parameter INHERITANCE_SIM.
The log file ```info.log``` is stored in the folder ```../results/contact_zones/<experiment_name>```.

*ContactZonesSimulator*.**logging_simulation()**

Log the information on the simulated features, which includes the number of features, global intensity, contact intensity, inherited intensity (equals contact intensity by default), global exposition (number of similar features), exposition in zone (number of similar features), exposition in family (number of similar features), upper bound for the number of zones.
The logging information is added to the file ```../results/contact_zones/<experiment_name>/info.log```.

A short usage example:

```
from simulation import ContactZonesSimulator

czs = ContactZonesSimulator()
czs.simulation()
czs.logging_setup()
czs.logging_simulation()
```

### *class* **MCMCSetup**(_log_path, results_path, network, features, families, label_results="default")