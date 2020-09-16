""" Class GeneralPlot

Inherits basic functions from Plot
Defines specific functions for general plots (not maps)
"""


from statistics import median
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from sbayes.plotting.plot_setup import Plot
from sbayes.postprocessing import compute_dic


class GeneralPlot(Plot):

    def __init__(self, simulated_data=False):

        # Load init function from the parent class Plot
        super().__init__(simulated_data=simulated_data)

    ####################################
    # Probability simplex, grid plot
    ####################################
    @staticmethod
    def get_corner_points(n, offset=0.5 * np.pi):
        """Generate corner points of a equal sided ´n-eck´."""
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
        return np.array([np.cos(angles), np.sin(angles)]).T

    @staticmethod
    def fill_outside(polygon, color, ax=None):
        """Fill the area outside the given polygon with ´color´.
        Args:
            polygon (np.array): The polygon corners in a numpy array.
                shape: (n_corners, 2)
            ax (plt.Axis): The pyplot axis.
            color (str or tuple): The fill color.
        """
        if ax is None:
            ax = plt.gca()

        n_corners = polygon.shape[0]
        i_left = np.argmin(polygon[:, 0])
        i_right = np.argmax(polygon[:, 0])

        # Find corners of bottom face
        i = i_left
        bot_x = [polygon[i, 0]]
        bot_y = [polygon[i, 1]]
        while i % n_corners != i_right:
            i += 1
            bot_x.append(polygon[i, 0])
            bot_y.append(polygon[i, 1])

        # Find corners of top face
        i = i_left
        top_x = [polygon[i, 0]]
        top_y = [polygon[i, 1]]
        while i % n_corners != i_right:
            i -= 1
            top_x.append(polygon[i, 0])
            top_y.append(polygon[i, 1])

        ymin, ymax = ax.get_ylim()
        plt.fill_between(bot_x, ymin, bot_y, color=color)
        plt.fill_between(top_x, ymax, top_y, color=color)

    # Transform weights into needed format
    def transform_weights(self, feature, b_in, gt=False):

        if not gt:
            universal_array = []
            contact_array = []
            inheritance_array = []
            sample_dict = self.results['weights']
            for key in sample_dict:
                split_key = key.split("_")
                if 'w' == split_key[0]:
                    if 'universal' == split_key[1] and str(feature) == split_key[2]:
                        universal_array = sample_dict[key][b_in:]
                    elif 'contact' == split_key[1] and str(feature) == split_key[2]:
                        contact_array = sample_dict[key][b_in:]
                    elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
                        inheritance_array = sample_dict[key][b_in:]

            sample = np.column_stack([universal_array, contact_array, inheritance_array]).astype(np.float)
            return sample

        else:

            true_universal = []
            true_contact = []
            true_inheritance = []

            true_dict = self.results['true_weights']

            for key in true_dict:
                split_key = key.split("_")
                if 'w' == split_key[0]:
                    if 'universal' == split_key[1] and str(feature) == split_key[2]:
                        true_universal = true_dict[key][b_in:]
                    elif 'contact' == split_key[1] and str(feature) == split_key[2]:
                        true_contact = true_dict[key][b_in:]
                    elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
                        true_inheritance = true_dict[key][b_in:]
            ground_truth = np.array([true_universal, true_contact, true_inheritance]).astype(np.float)
            return ground_truth

    def transform_probability_vectors(self, feature, parameter, b_in, gt=False):

        if not gt:
            if "alpha" in parameter:
                sample_dict = self.results['alpha']
            elif "beta" in parameter:
                sample_dict = self.results['beta']
            elif "gamma" in parameter:
                sample_dict = self.results['gamma']
            else:
                raise ValueError("parameter must be alpha, beta or gamma")
            p_dict = {}
            states = []

            for key in sample_dict:

                if str(feature + '_') in key and parameter in key:
                    state = str(key).rsplit('_', 1)[1]
                    p_dict[state] = sample_dict[key][b_in:]
                    states.append(state)

            sample = np.column_stack([p_dict[s] for s in p_dict]).astype(np.float)
            return sample, states
        else:
            if "alpha" in parameter:
                true_dict = self.results['true_alpha']
            elif "beta" in parameter:
                true_dict = self.results['true_beta']
            elif "gamma" in parameter:
                true_dict = self.results['true_gamma']
            else:
                raise ValueError("parameter must alpha, beta or gamma")

            true_prob = []

            for key in true_dict:
                if str(feature + '_') in key and parameter in key:
                    true_prob.append(true_dict[key])

            return np.array(true_prob).astype(np.float)

    # Sort weights by median contact
    def get_parameters(self, b_in, parameter="weights"):

        par = {}
        true_par = {}
        states = {}

        # get samples
        for i in self.results['feature_names']:
            if parameter == "weights":
                p = self.transform_weights(feature=i, b_in=b_in)
                par[i] = p

            elif "alpha" in parameter or "beta" in parameter or "gamma" in parameter:
                p, state = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in)

                par[i] = p
                states[i] = state

        # get ground truth
        if self.is_simulation:
            for i in self.results['feature_names']:
                if parameter == "weights":
                    true_p = self.transform_weights(feature=i, b_in=b_in, gt=True)
                    true_par[i] = true_p

                elif "alpha" in parameter or "beta" in parameter or "gamma" in parameter:
                    true_p = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in)
                    true_par[i] = true_p
        else:
            true_par = None
        return par, true_par, states

    def sort_by_weights(self, w):
        sort_by = {}
        for i in self.results['feature_names']:
            sort_by[i] = median(w[i][:, 1])
        ordering = sorted(sort_by, key=sort_by.get, reverse=True)
        return ordering

    # Probability simplex (for one feature)
    @staticmethod
    def plot_weights(samples, feature, true_weights=False, labels=None, ax=None, mean_weights=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_weights (np.array): true weight vectors (only for simulated data)
            labels (list[str]): Labels for each weight dimension.
            ax (plt.Axis): The pyplot axis.
            mean_weights (bool): Plot the mean of the weights?
        """

        if ax is None:
            ax = plt.gca()
        n_samples, n_weights = samples.shape

        # Compute corners
        corners = GeneralPlot.get_corner_points(n_weights)
        # Bounding box
        xmin, ymin = np.min(corners, axis=0)
        xmax, ymax = np.max(corners, axis=0)

        # Project the samples
        samples_projected = samples.dot(corners)

        # color map
        cmap = sns.cubehelix_palette(light=1, start=.5, rot=-.75, as_cmap=True)

        # Density and scatter plot
        plt.title(str(feature), loc='center', fontdict={'fontweight': 'bold', 'fontsize': 20})
        x = samples_projected.T[0]
        y = samples_projected.T[1]
        sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                    clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)
        plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)

        # Draw simplex and crop outside
        plt.fill(*corners.T, edgecolor='k', fill=False)
        GeneralPlot.fill_outside(corners, color='w', ax=ax)

        if true_weights:
            true_projected = true_weights.dot(corners)
            plt.scatter(*true_projected.T, color="#ed1696", lw=0, s=100, marker="*")

        if mean_weights:
            mean_projected = np.mean(samples, axis=0).dot(corners)
            plt.scatter(*mean_projected.T, color="#ed1696", lw=0, s=100, marker="o")

        if labels is not None:
            for xy, label in zip(corners, labels):
                xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 16})

        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout(0)
        plt.plot()

    @staticmethod
    def plot_probability_vectors(samples, feature, true_p=None, labels=None, ax=None, title=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_p (np.array): true probability vectors (only for simulated data)
            labels (list[str]): Labels for each weight dimension.
            title (bool): plot title
            ax (plt.Axis): The pyplot axis.
        """

        if ax is None:
            ax = plt.gca()
        n_samples, n_p = samples.shape
        # color map
        cmap = sns.cubehelix_palette(light=1, start=.5, rot=-.75, as_cmap=True)
        if n_p == 2:
            # plt.title(str(feature), loc='center', fontdict={'fontweight': 'bold', 'fontsize': 20})
            x = samples.T[0]
            y = np.zeros(n_samples)
            sns.distplot(x, rug=True, hist=False, kde_kws={"shade": True, "lw": 0, "clip": (0, 1)}, color="g",
                         rug_kws={"color": "k", "alpha": 0.01, "height": 0.03})
            # sns.kdeplot(x, shade=True, color="r", clip=(0, 1))
            # plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)
            # plt.axhline(y=0, color='k', linestyle='-', lw=0.5, xmin=0, xmax=1)

            ax.axes.get_yaxis().set_visible(False)
            # ax.annotate('', xy=(0, -0.5), xytext=(1, -0.1),
            #            arrowprops=dict(arrowstyle="-", color='b'))

            if true_p is not None:
                plt.scatter(true_p[0], 0, color="#ed1696", lw=0, s=100, marker="*")

            if labels is not None:
                for x, label in enumerate(labels):
                    plt.text(x, -0.5, label, ha='center', va='top', fontdict={'fontsize': 16})
            plt.plot([0, 1], [0, 0], c="k", lw=0.5)
            plt.xlim(0, 1)
            plt.axis('off')
            plt.tight_layout(0)

        elif n_p > 2:
            # Compute corners
            corners = GeneralPlot.get_corner_points(n_p)
            # Bounding box
            xmin, ymin = np.min(corners, axis=0)
            xmax, ymax = np.max(corners, axis=0)

            # Project the samples
            samples_projected = samples.dot(corners)

            # Density and scatter plot
            if title:
                plt.title(str(feature), loc='center', fontdict={'fontweight': 'bold', 'fontsize': 20})
            x = samples_projected.T[0]
            y = samples_projected.T[1]
            sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                        clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)
            plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.05)

            # Draw simplex and crop outside

            plt.fill(*corners.T, edgecolor='k', fill=False)
            GeneralPlot.fill_outside(corners, color='w', ax=ax)

            if true_p is not None:
                true_projected = true_p.dot(corners)
                plt.scatter(*true_projected.T, color="#ed1696", lw=0, s=100, marker="*")

            if labels is not None:
                for xy, label in zip(corners, labels):
                    xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                    plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 16})

            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout(0)

        plt.plot()

    # Find number of features
    # def find_num_features(self):
    #     for key in self.results['weights']:
    #         num = re.search('[0-9]+', key)
    #         if num:
    #             if int(num.group(0)) > self.number_features:
    #                 self.number_features = int(num.group(0))

    # Make a grid with all features (sorted by median contact)
    # By now we assume number of features to be 35; later this should be rewritten for any number of features
    # using find_num_features
    def plot_weights_grid(self, fname, labels=None, burn_in=0.4):

        print('Plotting weights...')
        burn_in = int(len(self.results['posterior']) * burn_in)

        weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)
        ordering = self.sort_by_weights(weights)

        n_plots = 4
        n_col = 4
        n_row = math.ceil(n_plots / n_col)

        fig, axs = plt.subplots(n_row, n_col, figsize=(15, 5))
        position = 1

        features = ordering[:n_plots]

        for f in features:
            plt.subplot(n_row, n_col, position)
            if self.is_simulation:
                self.plot_weights(weights[f], feature=f, true_weights=true_weights[f], labels=labels)
            else:
                self.plot_weights(weights[f], feature=f, labels=labels, mean_weights=True)
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        fig.savefig(self.path_plots + fname, dpi=400, format="pdf")

    # This is not changed yet
    def plot_probability_grid(self, fname, p_name="gamma_a1", burn_in=0.4):
        """Creates a ridge plot for parameters with two states

       Args:
           p_name (str): name of parameter vector (either alpha, beta_* or gamma_*)
           burn_in (float): fraction of the samples which should be discarded as burn_in
       """
        print('Plotting probabilities...')
        burn_in = int(len(self.results['posterior']) * burn_in)

        weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)

        ordering = self.sort_by_weights(weights)

        p, true_p, states = self.get_parameters(parameter=p_name, b_in=burn_in)

        n_plots = 4
        n_col = 4
        n_row = math.ceil(n_plots / n_col)

        fig, axs = plt.subplots(n_row, n_col, figsize=(15, 5))

        position = 1

        features = ordering[:n_plots]

        for f in features:
            plt.subplot(n_row, n_col, position)

            if self.is_simulation:
                self.plot_probability_vectors(p[f], feature=f, true_p=true_p[f], labels=states[f])
            else:
                self.plot_probability_vectors(p[f], feature=f, labels=states[f])
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        fig.savefig(self.path_plots + fname, dpi=400, format="pdf")

    def plot_dic(self, models, burn_in, simulated_data=False, threshold=False, fname='DICs'):
        """This function plots the dics. What did you think?
        Args:
            dics(dict): A dict of DICs from different models

        """
        print('Plotting DIC...')
        # if simulated_data:
        #     pp = get_plotting_params(plot_type="plot_dics_simulated")
        # else:
        #     pp = get_plotting_params(plot_type="plot_dics")
        #
        # plt.rcParams["axes.linewidth"] = pp['frame_width']

        fig, ax = plt.subplots(figsize=(20, 10))
        x = list(models.keys())
        y = []

        # Compute the DIC for each model
        for m in x:
            lh = models[m]['likelihood']
            dic = compute_dic(lh, burn_in)
            y.append(dic)

        ax.plot(x, y, lw=1, color='#000000', label='DIC')
        y_min, y_max = min(y), max(y)

        # round y min and y max to 1000 up and down, respectively
        n_digits = len(str(int(y_min))) - 1
        convertor = 10 ** (n_digits - 2)

        y_min = int(np.floor(y_min / convertor) * convertor)
        y_max = int(np.ceil(y_max / convertor) * convertor)

        ax.set_ylim([y_min, y_max])
        y_ticks = np.linspace(y_min, y_max, 6)
        ax.set_yticks(y_ticks)
        yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
        ax.set_yticklabels(yticklabels, fontsize=10)

        # if threshold:
        #     ax.axvline(x=threshold, lw=pp['line_thickness'], color=pp['color_burn_in'], linestyle='-')
        #     ypos_label = y_min + (y_max - y_min) / 2
        #     # ax.text(threshold, ypos_label, 'threshold', rotation=90, size=pp['fontsize'], color=pp['color_burn_in'])

        ax.set_ylabel('DIC', fontsize=10, fontweight='bold')

        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = list(range(1, len(x) + 1))
        ax.set_xticklabels(labels, fontsize=8)

        # x_min = 1
        # x_max = len(x)
        # x_ticks = np.linspace(x_min, x_max, len(x))
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels([int(x_tick) for x_tick in x_ticks], fontsize=8)

        ax.set_xlabel('Number of areas', fontsize=8, fontweight='bold')
        fig.savefig(self.path_plots + '/dic.pdf', dpi=400, format="pdf", bbox_inches='tight')
        # fig.savefig(f"{fname}.{pp['save_format']}", bbox_inches='tight', dpi=400, format=pp['save_format'])
