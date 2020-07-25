import numpy as np
from sbayes.plotting import plot_weights, plot_parameters_ridge

if __name__ == '__main__':
    labels = list(map(str, range(3)))
    weight_samples = np.random.dirichlet([1, 3, 5], size=(100,))
    plot_weights(weight_samples, labels)

    parameter_samples = np.random.dirichlet([1, 6], size=(100, ))
    plot_parameters_ridge(parameter_samples)

