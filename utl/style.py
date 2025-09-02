import matplotlib as mpl
import matplotlib.pyplot as plt
import os


GOLDEN_RATIO = 1.618033988749895


class MyStyle:
    def __init__(self):
        self.set_style()
        self.colors = [
            "black",
            "tab:blue",
            "tab:orange",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
            "tab:green",
            "darkblue",
            "darkred",
            "darkgreen",
            "darkorange",
            "darkviolet",
            "navy",
            "maroon",
            "forestgreen",
            "gold",
        ]
        self.colors2 = [
            "#376D9D",
            "#92DCE5",
            "#2CAC97",
            "#ABABAB",
            "#D64933",
            "#836055",
            "#E279A8",
            "#D9ED84",
            "#0AA536",
        ]
        self.annotation_size = 7
        self.dpi = 600

    def use_latex(self, switch):
        if switch:
            os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
            mpl.rcParams["text.usetex"] = True
            mpl.rcParams["font.family"] = "lmodern"
        else:
            mpl.rcParams["text.usetex"] = False
            mpl.rcParams["font.family"] = "DejaVu Sans"

    def set_style_poster(self):
        mpl.style.use("seaborn-v0_8-paper")

        # LaTex related
        mpl.rcParams["text.usetex"] = False

        # Font
        # mpl.rcParams['font.family'] = 'lmodern'
        mpl.rcParams["font.family"] = "DejaVu Sans"
        mpl.rcParams["font.size"] = 16
        # mpl.rcParams['mathtext.fontset'] = 'computer modern'
        # mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams["axes.titlesize"] = 16
        mpl.rcParams["axes.labelsize"] = 16

        # Lines and Markers
        mpl.rcParams["lines.linewidth"] = 3
        mpl.rcParams["lines.markersize"] = 4
        mpl.rcParams["lines.marker"] = "o"  # Add default marker
        mpl.rcParams["lines.markeredgewidth"] = 2  # Add marker edge width
        mpl.rcParams["lines.markeredgecolor"] = (
            "auto"  # Match marker edge color to line color
        )
        mpl.rcParams["lines.markerfacecolor"] = "none"  # Transparent marker face

        # Grid
        mpl.rcParams["grid.alpha"] = 0.3
        mpl.rcParams["grid.linestyle"] = "-"
        mpl.rcParams["axes.grid"] = False

        # Axes
        mpl.rcParams["xtick.labelsize"] = 16

        mpl.rcParams["ytick.labelsize"] = 16
        mpl.rcParams["ytick.major.pad"] = 3
        mpl.rcParams["axes.labelpad"] = 3
        mpl.rcParams["axes.linewidth"] = 1

        # width and length of the dashes
        mpl.rcParams["xtick.major.size"] = 4
        mpl.rcParams["xtick.minor.size"] = 2
        mpl.rcParams["xtick.major.width"] = 0.8
        mpl.rcParams["xtick.minor.width"] = 0.8
        mpl.rcParams["ytick.major.size"] = 4
        mpl.rcParams["ytick.minor.size"] = 2
        mpl.rcParams["ytick.major.width"] = 0.8
        mpl.rcParams["ytick.minor.width"] = 0.8

        # Ticks
        mpl.rcParams["xtick.top"] = True
        mpl.rcParams["ytick.right"] = True
        mpl.rcParams["xtick.direction"] = "in"
        mpl.rcParams["ytick.direction"] = "in"

        # Legend
        mpl.rcParams["legend.borderaxespad"] = 0.7
        mpl.rcParams["legend.fontsize"] = 14  # 8 default
        mpl.rcParams["legend.borderpad"] = 0.3  # Add spacing between axes and legend
        mpl.rcParams["legend.labelspacing"] = 0.3  # Add spacing between legend labels
        mpl.rcParams["legend.markerscale"] = (
            1.0  # Add spacing between legend line and legend text
        )
        mpl.rcParams["legend.handletextpad"] = (
            0.3  # Add spacing between legend line and legend text
        )

        # Size
        width = 6.2
        mpl.rcParams["figure.figsize"] = width, width / GOLDEN_RATIO

        # Save figure
        mpl.rcParams["savefig.transparent"] = False
        mpl.rcParams["savefig.dpi"] = 600
        mpl.rcParams["savefig.format"] = "png"

    def set_style(self):
        mpl.style.use("seaborn-v0_8-paper")

        # LaTex related
        #mpl.rcParams["text.usetex"] = False

        # Font
        # mpl.rcParams['font.family'] = 'lmodern'
        mpl.rcParams["font.family"] = "DejaVu Sans"
        mpl.rcParams["font.size"] = 10
        # mpl.rcParams['mathtext.fontset'] = 'computer modern'
        # mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams["axes.titlesize"] = 10
        mpl.rcParams["axes.labelsize"] = 10

        # Lines and Markers
        mpl.rcParams["lines.linewidth"] = 1.5
        mpl.rcParams["lines.markersize"] = 4
        mpl.rcParams["lines.marker"] = "o"  # Add default marker
        mpl.rcParams["lines.markeredgewidth"] = 1  # Add marker edge width
        mpl.rcParams["lines.markeredgecolor"] = (
            "auto"  # Match marker edge color to line color
        )
        mpl.rcParams["lines.markerfacecolor"] = "none"  # Transparent marker face

        # Grid
        mpl.rcParams["grid.alpha"] = 0.3
        mpl.rcParams["grid.linestyle"] = "--"
        mpl.rcParams["axes.grid"] = False
        mpl.rcParams["grid.linewidth"] = 0.5

        # Axes
        mpl.rcParams["xtick.labelsize"] = 10

        mpl.rcParams["ytick.labelsize"] = 10
        mpl.rcParams["ytick.major.pad"] = 3
        mpl.rcParams["axes.labelpad"] = 3
        mpl.rcParams["axes.linewidth"] = 0.5

        # width and length of the dashes
        mpl.rcParams["xtick.major.size"] = 4
        mpl.rcParams["xtick.minor.size"] = 2
        mpl.rcParams["xtick.major.width"] = 0.8
        mpl.rcParams["xtick.minor.width"] = 0.8
        mpl.rcParams["ytick.major.size"] = 4
        mpl.rcParams["ytick.minor.size"] = 2
        mpl.rcParams["ytick.major.width"] = 0.8
        mpl.rcParams["ytick.minor.width"] = 0.8

        # Ticks
        mpl.rcParams["xtick.top"] = True
        mpl.rcParams["ytick.right"] = True
        mpl.rcParams["xtick.direction"] = "in"
        mpl.rcParams["ytick.direction"] = "in"

        # Legend
        mpl.rcParams["legend.borderaxespad"] = 0.7
        mpl.rcParams["legend.fontsize"] = 8  # 8 default
        mpl.rcParams["legend.title_fontsize"] = 8
        mpl.rcParams["legend.borderpad"] = 0.3  # Add spacing between axes and legend
        mpl.rcParams["legend.labelspacing"] = 0.3  # Add spacing between legend labels
        mpl.rcParams["legend.markerscale"] = (
            1.0  # Add spacing between legend line and legend text
        )
        mpl.rcParams["legend.handletextpad"] = (
            0.3  # Add spacing between legend line and legend text
        )

        # Size
        width = 6.2
        mpl.rcParams["figure.figsize"] = width, width / GOLDEN_RATIO

        # Save figure
        mpl.rcParams["savefig.transparent"] = False
        mpl.rcParams["savefig.dpi"] = 600
        mpl.rcParams["savefig.format"] = "png"

    def set_width(self, width=3, ratio=1):
        """
        Set the width of the figure. The height is calculated using the golden ratio.
        :param width: Width of the figure in inches
        :param ratio: Ratio of the width to the height times the golden ratio
        """
        ratio = ratio * GOLDEN_RATIO
        mpl.rcParams["figure.figsize"] = width, width / ratio

    def set_size(self, width=3, height=3):
        """
        Set the size of the figure.
        :param width: Width of the figure in inches
        :param height: Height of the figure in inches
        """
        mpl.rcParams["figure.figsize"] = width, height

    def set_width_single_subplot(self):
        self.set_width(4, 1)
        return self

    def set_width_double_subplot(self):
        self.set_width(6, 2)
        return self

    def get_figsize(self):
        return mpl.rcParams["figure.figsize"][0], mpl.rcParams["figure.figsize"][1]

    def get_color(self, idx, mode="default", num_colors=10):
        """
        Get the colors for the plot.
        :param mode: "default" or "2" for the second set of colors
        :return: List of colors
        """
        if mode == "default":
            return self.colors[idx % len(self.colors)]
        elif mode == "2":
            return self.colors2[idx % len(self.colors2)]
        else:
            try:
                cmap = plt.get_cmap(mode)
                colors = [cmap(i / max(num_colors - 1, 1)) for i in range(num_colors)]
                return colors[idx % len(colors)]
            except ValueError:
                print(f"Error: The colormap '{mode}' is not available. Falling back to 'viridis'.")
                cmap = plt.get_cmap("viridis")
                colors = [cmap(i / max(num_colors - 1, 1)) for i in range(num_colors)]
                return colors[idx % len(colors)]

    def set_scientific_notation(self, ax, axis="both"):
        ax.ticklabel_format(axis=axis, style="sci", scilimits=(4, 4))

    def add_subplot_label(self, fig, ax, label, width=0.2, color="black"):
        """
        : width: Width of the label in inches
        """

        x_limits = ax.get_xlim()
        ax.set_xlim(x_limits)

        fig.canvas.draw()

        # Get the DPI (dots per inch) of the figure
        dpi = fig.dpi

        # Get the bounding box of the axes in display units
        bbox0 = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # Calculate the width and height of the axes in inches
        cw, ch = bbox0.width, bbox0.height

        # print(f"Axis width in inches: {cw}")
        # print(f"Axis height in inches: {ch}")

        bbox_x, bbox_y = 0, 1  # Top left corner in axis coordinates
        bbox_width, bbox_height = (
            width / cw,
            width / ch,
        )  # Width and height in axis coordinates

        rect = mpl.patches.Rectangle(
            (bbox_x, bbox_y - bbox_height),
            bbox_width,
            bbox_height,
            transform=ax.transAxes,
            color=color,
            clip_on=True,
            zorder=100,
        )
        ax.add_patch(rect)

        ax.text(
            bbox_x + bbox_width / 2,
            bbox_y - bbox_height / 1.8,
            label,
            transform=ax.transAxes,
            fontsize=16,
            color="white",
            fontweight="bold",
            verticalalignment="center",
            horizontalalignment="center",
            zorder=101,
        )

    def add_time_ax_to_plot(self, ax, res):
        ax = ax.twiny()  # Create a second x-axis sharing the same y-axis
        ax.plot(
            res.df["time_axis"], res.df["iphoto_man"], alpha=0
        )  # Invisible plot to align the secondary x-axis
        ax.set_xlabel("Measurement Time (h)")
        ax.xaxis.set_label_coords(
            0.5, 1.22
        )  # Adjust the second parameter to increase or decrease the padding

    def add_reference_text(self, plt, ref, dir=False, location=(0.98, 0.97)):
        """
        Adds a reference text inside the axes, near the upper axis.

        Args:
            plt: matplotlib.pyplot object
            ref: string with the reference text
        """
        if dir:
            ref = "source: " + "/".join(ref.split("/")[:-1])
        ax = plt.gca()  # Get current axes
        ax.text(
            location[0],
            location[1],
            ref,
            ha="right",
            va="top",
            color="grey",
            fontsize=self.annotation_size,
            transform=ax.transAxes,
        )  # Using axes coordinates
        return plt


    def add_reference_text_ax(self, ax, ref, location=(0.98, 0.97)):
        """
        Adds a reference text inside the axes, near the upper axis.

        Args:
            ax: matplotlib.axes.Axes object
            ref: string with the reference text
        """
        ax.text(
            location[0],
            location[1],
            ref,
            ha="right",
            va="top",
            color="grey",
            fontsize=self.annotation_size,
            transform=ax.transAxes,
        )

    def save_plot_for_thesis(self, fig, filename, subdir=None, format='pdf'):
        """
        Save the current figure to the thesis plot directory.
        """
        THESIS_PLOT_PATH = "../WORKSPACE/THESIS_PLOTS"
        width = mpl.rcParams["figure.figsize"][0]
        filename = f"{filename}_w{width:.2f}"
        if not os.path.exists(THESIS_PLOT_PATH):
            os.makedirs(THESIS_PLOT_PATH)
        if subdir:
            THESIS_PLOT_PATH = os.path.join(THESIS_PLOT_PATH, subdir)
            if not os.path.exists(THESIS_PLOT_PATH):
                os.makedirs(THESIS_PLOT_PATH)

        fig.savefig(os.path.join(THESIS_PLOT_PATH, filename) + "." + format, bbox_inches='tight', dpi=600, format=format)
        print(f"Plot saved to {os.path.join(THESIS_PLOT_PATH, filename) + '.' + format}")

class PresStyle(MyStyle):
    def __init__(self):
        super().__init__()
        mpl.rcParams["lines.linewidth"] = 1.5

        # set figsize
        # width = 6.9
        # mpl.rcParams['figure.figsize'] = width, width / scipy.constants.golden_ratio

        # Font
        mpl.rcParams["font.size"] = 14
        mpl.rcParams["axes.titlesize"] = 14
        mpl.rcParams["axes.labelsize"] = 14
        mpl.rcParams["xtick.labelsize"] = 14
        mpl.rcParams["ytick.labelsize"] = 14
        mpl.rcParams["ytick.major.pad"] = 4
        mpl.rcParams["axes.labelpad"] = 4
        mpl.rcParams["axes.linewidth"] = 0.5

        mpl.rcParams["legend.borderaxespad"] = 0.7
        mpl.rcParams["legend.fontsize"] = 10
        mpl.rcParams["legend.borderpad"] = 0.3  # Add spacing between axes and legend
        mpl.rcParams["legend.labelspacing"] = 0.3  # Add spacing between legend labels
        mpl.rcParams["legend.markerscale"] = (
            1.0  # Add spacing between legend line and legend text
        )
        mpl.rcParams["legend.handletextpad"] = (
            0.3  # Add spacing between legend line and legend text
        )
