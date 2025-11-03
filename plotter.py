import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import norm

plt.rcParams.update({
    'font.size': 15,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.minor.size': 10,
    'xtick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.major.width': 3,
    'ytick.minor.width': 3,
    'axes.linewidth': 3,
    'figure.max_open_warning': 200,
    'lines.linewidth': 5
})

class Plotter:
    def __init__(self, print_dir='', end_name=''):
        """
        Parameters
        ----------
        print_dir : str
            Directory where plots will be saved.
        end_name : str
            Optional suffix appended to output file names.
        """
        self.print_dir = print_dir
        self.end_name = end_name

    def plotTrainLoss(self, tracker):
        """
        Plot training and validation loss curves.

        Parameters
        ----------
        tracker : object
            Object with attributes 'train_losses' and 'val_losses', containing per-epoch loss values.
        """
        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        plt.figure(figsize=(20, 20))
        plt.plot(train_losses, label='Train', color='royalblue')
        plt.plot(val_losses, label='Test', color='firebrick')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}/loss_{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_diff(self, preds, targets):
        """
        Plot histograms of prediction errors (predicted - target) for each feature,
        and fit a Gaussian in feature-specific range.

        """
        feature_names = ['avgWire', 'slope']
        xlims = [(-5, 5), (-0.2, 0.2)]
        fit_ranges = [(-0.5, 0.5), (-0.025, 0.025)]

        for i in range(2):
            plt.figure(figsize=(8, 6))
            diff = preds[:, i] - targets[:, i]

            # Plot histogram of counts
            counts, bins, _ = plt.hist(diff, bins=500, density=False, alpha=0.6)
            plt.xlabel(f"Diff ({feature_names[i]})")
            plt.ylabel("Counts")
            plt.title(f"Diff: {feature_names[i]}")
            plt.xlim(*xlims[i])

            # Gaussian fit in restricted window
            fit_min, fit_max = fit_ranges[i]
            mask = (diff >= fit_min) & (diff <= fit_max)
            diff_fit = diff[mask]
            if len(diff_fit) > 5:
                mu, std = norm.fit(diff_fit)

                # Scale Gaussian to histogram counts
                bin_width = bins[1] - bins[0]
                x = np.linspace(bins[0], bins[-1], 500)
                p = norm.pdf(x, mu, std) * len(diff_fit) * bin_width
                plt.plot(x, p, 'r', linewidth=2)

                plt.text(
                    0.95, 0.95,
                    f"$\\mu={mu:.3f}$\n$\\sigma={std:.3f}$",
                    transform=plt.gca().transAxes,
                    ha="right", va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
                )

            outname = f"{self.print_dir}/diff_{feature_names[i]}_{self.end_name}.png"
            plt.savefig(outname, bbox_inches='tight')
            plt.close()

    def plot_pred_target(self, preds, targets):
        """
        Plot 2D histograms comparing predicted vs target values for each feature.
        """
        feature_names = ['avgWire', 'slope']
        ranges = [[[0, 112], [0, 112]],  # avgWire
                  [[-1, 1], [-1, 1]]]  # slope

        for i in range(2):
            plt.figure(figsize=(8, 6))
            x_range, y_range = ranges[i]

            plt.hist2d(targets[:, i], preds[:, i], bins=(300, 300),
                       range=[x_range, y_range], cmap='viridis', norm=LogNorm(vmin=1))
            plt.colorbar(label='Counts')

            plt.xlabel(f"Target ({feature_names[i]})")
            plt.ylabel(f"Prediction ({feature_names[i]})")
            plt.title(f"Prediction vs Target ({feature_names[i]})")
            plt.plot(x_range, y_range, 'r--', label='Ideal Prediction')
            plt.legend()
            plt.axis('equal')

            outname = f"{self.print_dir}/pred_target_{feature_names[i]}_{self.end_name}.png"
            plt.savefig(outname, bbox_inches='tight')
            plt.close()
