""" Class GeneralPlot

Inherits basic functions from Plot
Defines specific functions for general plots (not maps)
"""


from statistics import median
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import matplotlib.ticker as mtick
from matplotlib.ticker import AutoMinorLocator

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
                        true_universal = true_dict[key]
                    elif 'contact' == split_key[1] and str(feature) == split_key[2]:
                        true_contact = true_dict[key]
                    elif 'inheritance' == split_key[1] and str(feature) == split_key[2]:
                        true_inheritance = true_dict[key]
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
                    true_p = self.transform_probability_vectors(feature=i, parameter=parameter, b_in=b_in, gt=True)
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
    def plot_weights(samples, feature, title, true_weights=None,
                     labels=None, ax=None, mean_weights=False):
        """Plot a set of weight vectors in a 2D representation of the probability simplex.

        Args:
            samples (np.array): Sampled weight vectors to plot.
            feature (str): Name of the feature for which weights are being plotted
            true_weights (np.array): the ground truth weights
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
        if title:
            plt.text(-0.7, 0.6, str(feature), fontdict={'fontweight': 'bold', 'fontsize': 12})
        x = samples_projected.T[0]
        y = samples_projected.T[1]
        sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                    clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)
        #plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)

        # Draw simplex and crop outside
        plt.fill(*corners.T, edgecolor='k', fill=False)
        GeneralPlot.fill_outside(corners, color='w', ax=ax)

        if true_weights is not None:
            true_weights_projected = true_weights.dot(corners)
            plt.scatter(*true_weights_projected.T, color="#ed1696", lw=0, s=200, marker="*")

        if mean_weights:

            mean_projected = np.mean(samples, axis=0).dot(corners)
            plt.scatter(*mean_projected.T, color="#ed1696", lw=0, s=200, marker="o")

        if labels is not None:
            for xy, label in zip(corners, labels):

                xy *= 1.08  # Stretch, s.t. labels don't overlap with corners
                plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 12})

        plt.xlim(xmin-0.1, xmax+0.1)
        plt.ylim(ymin-0.1, ymax+0.1)
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
            x = samples.T[1]
            sns.distplot(x, rug=True, hist=False, kde_kws={"shade": True, "lw": 0, "clip": (0, 1)}, color="g",
                         rug_kws={"color": "k", "alpha": 0.01, "height": 0.03})
            # sns.kdeplot(x, shade=True, color="r", clip=(0, 1))
            # plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.2)
            # plt.axhline(y=0, color='k', linestyle='-', lw=0.5, xmin=0, xmax=1)

            ax.axes.get_yaxis().set_visible(False)
            # ax.annotate('', xy=(0, -0.5), xytext=(1, -0.1),
            #            arrowprops=dict(arrowstyle="-", color='b'))

            if true_p is not None:
                plt.scatter(true_p[1], 0, color="#ed1696", lw=0, s=200, marker="*")

            if labels is not None:
                for x, label in enumerate(labels):
                    if x == 0:
                        x = -0.05
                    if x == 1:
                        x = 1.05
                    plt.text(x, 0.1, label, ha='center', va='top', fontdict={'fontsize': 12})
            if title:
                plt.text(0.3, 4, str(feature), fontsize=12, fontweight='bold')

            plt.plot([0, 1], [0, 0], c="k", lw=0.5)

            #y_max = ax.get_ylim()[1]
            ax.axes.set_ylim([-0.2, 5])
            ax.axes.set_xlim([0, 1])

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
                plt.text(-0.8, 0.8, str(feature), fontsize=12, fontweight='bold')
                #plt.title(str(feature), loc='left', fontdict={'fontweight': 'bold', 'fontsize': 16})
            x = samples_projected.T[0]
            y = samples_projected.T[1]
            sns.kdeplot(x, y, shade=True, shade_lowest=True, cut=30, n_levels=100,
                        clip=([xmin, xmax], [ymin, ymax]), cmap=cmap)
            # plt.scatter(x, y, color='k', lw=0, s=1, alpha=0.05)

            # Draw simplex and crop outside

            plt.fill(*corners.T, edgecolor='k', fill=False)
            GeneralPlot.fill_outside(corners, color='w', ax=ax)

            if true_p is not None:

                true_projected = true_p.dot(corners)
                plt.scatter(*true_projected.T, color="#ed1696", lw=0, s=200, marker="*")

            if labels is not None:
                for xy, label in zip(corners, labels):
                    xy *= 1.1  # Stretch, s.t. labels don't overlap with corners
                    plt.text(*xy, label, ha='center', va='center', fontdict={'fontsize': 12})

            plt.xlim(xmin - 0.1, xmax + 0.1)
            plt.ylim(ymin - 0.1, ymax + 0.1)
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
    def plot_weights_grid(self, file_name, file_format="pdf"):

        print('Plotting weights...')
        burn_in = int(len(self.results['posterior']) * self.config['weights_plot']['burn_in'])

        weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)
        # ordering = self.sort_by_weights(weights)
        ordering = self.results['feature_names']

        n_plots = self.config['weights_plot']['k_best']
        n_col = self.config['weights_plot']['n_columns']
        n_row = math.ceil(n_plots / n_col)
        width = self.config['weights_plot']['output']['fig_width_subplot']
        height = self.config['weights_plot']['output']['fig_height_subplot']
        fig, axs = plt.subplots(n_row, n_col, figsize=(width*n_col, height*n_row))
        position = 1

        features = ordering[:n_plots]
        labels = self.config['weights_plot']['labels']
        n_empty = n_row * n_col - n_plots

        for e in range(1, n_empty + 1):
            axs[-1, -e].axis('off')

        for f in features:
            plt.subplot(n_row, n_col, position)

            if self.is_simulation:
                self.plot_weights(weights[f], feature=f, title=self.config['weights_plot']['title'],
                                  true_weights=true_weights[f], labels=labels)
            else:
                self.plot_weights(weights[f], feature=f, title=self.config['weights_plot']['title'],
                                  labels=labels, mean_weights=False)
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=self.config['weights_plot']['output']['spacing_horizontal'],
                            hspace=self.config['weights_plot']['output']['spacing_vertical'])

        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=400, format=file_format)
        plt.close(fig)

    # This is not changed yet
    def plot_probability_grid(self, file_name, file_format="pdf"):
        """Creates a ridge plot for parameters with two states

       Args:
           file_name (str): name of the output file
           file_format (str): output file format
       """
        print('Plotting probabilities...')
        burn_in = int(len(self.results['posterior']) * self.config['probabilities_plot']['burn_in'])

        n_plots = self.config['probabilities_plot']['k']
        n_col = self.config['probabilities_plot']['n_columns']
        n_row = math.ceil(n_plots / n_col)
        width = self.config['probabilities_plot']['output']['fig_width_subplot']
        height = self.config['probabilities_plot']['output']['fig_height_subplot']

        #weights, true_weights, _ = self.get_parameters(parameter="weights", b_in=burn_in)
        #ordering = self.sort_by_weights(weights)

        p, true_p, states = self.get_parameters(parameter=self.config['probabilities_plot']['parameter'], b_in=burn_in)
        fig, axs = plt.subplots(n_row, n_col, figsize=(width*n_col, height*n_row), )

        features = self.results['feature_names']
        n_empty = n_row * n_col - n_plots

        for e in range(1, n_empty+1):
            axs[-1, -e].axis('off')

        position = 1

        for f in features[0:n_plots]:
            plt.subplot(n_row, n_col, position)

            if self.is_simulation:
                self.plot_probability_vectors(p[f], feature=f, true_p=true_p[f], labels=states[f],
                                              title=self.config['probabilities_plot']['title'])
            else:

                self.plot_probability_vectors(p[f], feature=f, labels=states[f],
                                              title=self.config['probabilities_plot']['title'])
            print(position, "of", n_plots, "plots finished")
            position += 1

        plt.subplots_adjust(wspace=self.config['probabilities_plot']['output']['spacing_horizontal'],
                            hspace=self.config['probabilities_plot']['output']['spacing_vertical'])
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=400,
                    format=file_format)
        plt.close(fig)

    def plot_dic(self, models, file_name, file_format="pdf"):
        """This function plots the dics. What did you think?
        Args:
            file_name (str): name of the output file
            file_format (str): output file format
            models(dict): A dict of different models for which the DIC is evaluated
        """
        print('Plotting DIC...')
        width = self.config['dic_plot']['output']['fig_width']
        height = self.config['dic_plot']['output']['fig_height']

        fig, ax = plt.subplots(figsize=(width, height))
        x = list(models.keys())
        y = []

        # Compute the DIC for each model
        for m in x:
            lh = models[m]['likelihood']
            dic = compute_dic(lh, self.config['dic_plot']['burn_in'])
            y.append(dic)

        # Limits
        ax.plot(x, y, lw=2, color='#000000', label='DIC')
        y_min, y_max = min(y), max(y)
        y_range = y_max - y_min

        x_min = 0
        x_max = len(x)-1

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - y_range * 0.1, y_max + y_range * 0.1])

        # Labels and ticks
        ax.set_xlabel('Number of areas', fontsize=16, fontweight='bold')
        ax.set_ylabel('DIC', fontsize=16, fontweight='bold')

        labels = list(range(1, len(x) + 1))
        ax.set_xticklabels(labels, fontsize=14)

        y_ticks = np.linspace(y_min, y_max, 6)
        ax.set_yticks(y_ticks)
        yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
        ax.set_yticklabels(yticklabels, fontsize=14)
        try:
            if self.config['dic_plot']['true_n'] is not None:
                pos_true_model = [idx for idx, val in enumerate(x) if val == self.config['dic_plot']['true_n']][0]
                color_burn_in = 'grey'
                ax.axvline(x=pos_true_model, lw=2, color=color_burn_in, linestyle='--')
                ypos_label = y_min + y_range * 0.15
                plt.text(pos_true_model - 0.25, ypos_label, 'Simulated areas', rotation=90, size=16,
                         color=color_burn_in)
        except KeyError:
            pass
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format, dpi=400, bbox_inches='tight',
                    format=file_format)

    def plot_trace(self, file_name="trace", show_every_k_sample=1, file_format="pdf"):
        """
        Function to plot the trace of a parameter
        Args:
            file_name (str): a path followed by a the name of the file
            file_format (str): format of the output file
            show_every_k_sample (int): show every 1, 1+k,1+2k sample and skip over the rest
        """
        # For Olga: change all three parameters to config file entries
        plt.rcParams["axes.linewidth"] = 1
        fig, ax = plt.subplots(figsize=(self.config['plot_trace']['output']['fig_width'],
                                        self.config['plot_trace']['output']['fig_height']))
        parameter = self.config['plot_trace']['parameter']
        if parameter == 'recall_and_precision':
            y = self.results['recall'][::show_every_k_sample]
            y2 = self.results['precision'][::show_every_k_sample]
            x = self.results['sample_id'][::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color=self.config['plot_trace']['color'][0], label="recall")
            ax.plot(x, y2, lw=0.5, color=self.config['plot_trace']['color'][1], label="precision")
            y_min = 0
            y_max = 1

        else:
            try:
                y = self.results[parameter][::show_every_k_sample]

            except KeyError:
                raise ValueError("Cannot compute trace. " + self.config['plot_trace']['parameter']
                                 + " is not a valid parameter.")

            x = self.results['sample_id'][::show_every_k_sample]
            ax.plot(x, y, lw=0.5, color=self.config['plot_trace']['color'][0], label=parameter)
            y_min, y_max = min(y), max(y)

        y_range = y_max - y_min
        x_min, x_max = 0, x[-1]

        if self.config['plot_trace']['ground_truth']['add']:
            ground_truth_parameter = 'true_' + parameter
            y_gt = self.results[ground_truth_parameter]
            ax.axhline(y=y_gt, xmin=x[0], xmax=x[-1], lw=3, color='#fdbf6f',
                       linestyle='-',
                       label='ground truth')
            y_min, y_max = [min(y_min, y_gt), max(y_max, y_gt)]

        # Show burn-in in plot
        end_bi = math.ceil(x[-1] * self.config['plot_trace']['burn_in'])
        end_bi_label = math.ceil(x[-1] * (self.config['plot_trace']['burn_in'] - 0.04))

        color_burn_in = 'grey'
        ax.axvline(x=end_bi, lw=2, color=color_burn_in, linestyle='--')
        ypos_label = y_min + y_range * 0.05
        plt.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=16, color=color_burn_in)

        # Ticks and labels
        n_ticks = 6 if int(self.config['plot_trace']['burn_in'] * 100) % 20 == 0 else 12
        x_ticks = np.linspace(x_min, x_max, n_ticks)
        x_ticks = [round(t, -5) for t in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=14)

        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))
        ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')

        y_ticks = np.linspace(y_min, y_max, 5)
        ax.set_yticks(y_ticks)
        y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
        ax.set_yticklabels(y_ticklabels, fontsize=14)

        # Limits
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min - y_range * 0.01, y_max + y_range * 0.01])

        # Legend
        ax.legend(loc=4, prop={'size': 14}, frameon=False)

        # Save
        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format,
                    dpi=400, format=file_format, bbox_inches='tight')
        plt.close(fig)

    def plot_trace_lh_prior(self, burn_in=0.2, fname="trace", show_every_k_sample=1, lh_lim=None, prior_lim=None):
        fig, ax1 = plt.subplots(figsize=(10, 8))

        lh = self.results['likelihood'][::show_every_k_sample]
        prior = self.results['prior'][::show_every_k_sample]
        x = self.results['sample_id'][::show_every_k_sample]

        # Plot lh on axis 1
        ax1.plot(x, lh, lw=0.5, color='#e41a1c', label='likelihood')

        # Plot prior on axis 2
        ax2 = ax1.twinx()
        ax2.plot(x, prior, lw=0.5, color='dodgerblue', label='prior')

        x_min, x_max = 0, x[-1]
        if lh_lim is None:
            lh_min, lh_max = min(lh), max(lh)
        else:
            lh_min, lh_max = lh_lim
        lh_range = lh_max - lh_min

        if prior_lim is None:
            prior_min, prior_max = min(lh), max(lh)
        else:
            prior_min, prior_max = prior_lim
        prior_range = prior_max - prior_min

        # Labels and ticks
        n_ticks = 6 if int(burn_in * 100) % 20 == 0 else 12
        x_ticks = np.linspace(x_min, x_max, n_ticks)
        x_ticks = [round(t, -5) for t in x_ticks]
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=6)

        f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(g))
        ax1.set_xlabel('Iteration', fontsize=8, fontweight='bold')

        lh_ticks = np.linspace(lh_min, lh_max, 6)
        ax1.set_yticks(lh_ticks)
        lh_ticklabels = [f'{lh_tick:.0f}' for lh_tick in lh_ticks]
        ax1.set_yticklabels(lh_ticklabels, fontsize=6, color='#e41a1c')
        ax1.set_ylabel('log-likelihood', fontsize=8, fontweight='bold', color='#e41a1c')

        prior_ticks = np.linspace(prior_min, prior_max, 6)
        ax2.set_yticks(prior_ticks)
        prior_ticklabels = [f'{prior_tick:.0f}' for prior_tick in prior_ticks]
        ax2.set_yticklabels(prior_ticklabels, fontsize=8, color='dodgerblue')
        ax2.set_ylabel('prior', fontsize=8, fontweight='bold', color='dodgerblue')

        # Show burn-in in plot
        end_bi = math.ceil(x[-1] * burn_in)
        end_bi_label = math.ceil(x[-1] * (burn_in - 0.03))

        color_burn_in = 'grey'
        ax1.axvline(x=end_bi, lw=1, color=color_burn_in, linestyle='--')
        ypos_label = prior_min + prior_range * 0.15
        plt.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=10, color=color_burn_in)

        # Limits
        ax1.set_ylim([lh_min - lh_range * 0.1, lh_max + lh_range * 0.1])
        ax2.set_ylim([prior_min - prior_range * 0.1, prior_max + prior_range * 0.1])

        # Save
        fig.savefig(self.path_plots + fname + '.pdf', dpi=400, format="pdf", bbox_inches='tight')
        plt.close(fig)

    def plot_recall_precision_over_all_models(self, models, file_name, file_format="pdf"):
        width = self.config['recall_precision_over_all_models_plot']['output']['fig_width']
        height = self.config['recall_precision_over_all_models_plot']['output']['fig_height']
        fig, ax = plt.subplots(figsize=(width, height))

        recall = []
        precision = []

        # Retrieve recall and precision for every model and plot
        for m in list(models.keys()):
            recall.extend(models[m]['recall'])
            precision.extend(models[m]['precision'])

        x = list(range(len(recall)))
        ax.plot(x, recall, lw=0.5, color='#e41a1c', label='recall')
        ax.plot(x, precision, lw=0.5, color='dodgerblue', label='precision')

        # Limits
        ax.set_ylim(bottom=0)
        x_min, x_max = 0, len(recall)
        y_min, y_max, y_step = 0, 1, 0.2
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max + 0.1])

        # Labels
        n_models = len(list(models.keys()))
        n_ticks = n_models + 1
        x_ticks = np.linspace(x_min, x_max, n_ticks)

        x_ticks_offset = x_ticks[1] / 2
        x_ticks = [x_tick - x_ticks_offset for x_tick in x_ticks if x_tick > 0]
        ax.set_xticks(x_ticks)
        x_ticklabels = [f'{x_ticklabel:.0f} areas' for x_ticklabel in np.linspace(1, n_models, n_models)]
        x_ticklabels[0] = '1 area'
        ax.set_xticklabels(x_ticklabels, fontsize=14)

        minor_locator = AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(which='minor', axis='x', color='#000000', linestyle='-')
        ax.set_axisbelow(True)

        y_ticks = np.arange(y_min, y_max + y_step, y_step)
        ax.set_yticks(y_ticks)
        y_ticklabels = [f'{y_tick:.1f}' for y_tick in y_ticks]
        y_ticklabels[0] = '0'
        ax.set_yticklabels(y_ticklabels, fontsize=14)

        ax.legend(loc=4, prop={'size': 14}, frameon=True, framealpha=1, facecolor='#ffffff',
                  edgecolor='#ffffff')

        fig.savefig(self.path_plots + '/' + file_name + '.' + file_format,
                    dpi=400, format=file_format, bbox_inches='tight')
        plt.close(fig)
