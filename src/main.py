import matplotlib.pyplot as plt
import numpy as np

from data import get_RSI, get_top_vol_coins

FIGURE_SIZE = (12, 10)
BACKGROUND_COLOR = "#0d1117"
RANGES = {
    "Overbought": (70, 100),
    "Strong": (60, 70),
    "Neutral": (40, 60),
    "Weak": (30, 40),
    "Oversold": (0, 30),
}
COLORS_LABELS = {
    "Oversold": "#1d8b7a",
    "Weak": "#144e48",
    "Neutral": "#0d1117",
    "Strong": "#681f28",
    "Overbought": "#c32e3b",
}
SCATTER_COLORS = {
    "Oversold": "#1e9884",
    "Weak": "#165952",
    "Neutral": "#78797a",
    "Strong": "#79212c",
    "Overbought": "#cf2f3d",
}


def get_color_for_rsi(rsi_value):
    for label, (low, high) in RANGES.items():
        if low <= rsi_value < high:
            return SCATTER_COLORS[label]
    return None


def plot_rsi_heatmap(num_coins: int = 100):
    top_vol = get_top_vol_coins(num_coins)
    data = get_RSI(top_vol)

    # Create lists of labels and RSI values
    labels = list(data.keys())
    rsi_values = list(data.values())

    # Calculate the average RSI value
    average_rsi = np.mean(rsi_values)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(
        BACKGROUND_COLOR
    )  # Set the background color of the figure to black
    ax.set_facecolor(BACKGROUND_COLOR)  # Set the background color of the axes to black

    # Define the color for each RSI range
    color_map = []
    for k in RANGES:
        color_map.append((*RANGES[k], COLORS_LABELS[k], k))

    # Fill the areas with the specified colors and create custom legend
    for start, end, color, label in color_map:
        ax.fill_between([0, len(labels) + 2], start, end, color=color, alpha=0.35)

        # TODO: add text the right of the plot with the label (overbought, etc.)

    # Plot each point with a white border for visibility
    for i, label in enumerate(labels):
        # These are the dots on the plot
        ax.scatter(
            i + 1,
            rsi_values[i],
            color=get_color_for_rsi(rsi_values[i]),
            # edgecolor="black",
            s=100,
        )
        # Add the symbol text
        ax.annotate(
            label,
            (i + 1, rsi_values[i]),
            color="white",
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Draw the average RSI line and add the annotation
    ax.axhline(xmin=0, xmax=1, y=average_rsi, color="orange", linestyle="--")
    ax.text(
        len(labels),  # Increase to move the text to the right
        average_rsi,
        f"AVG RSI: {average_rsi:.2f}",
        color="orange",
        va="bottom",
        ha="center",
        fontsize=16,
    )

    # Set the color of the tick labels to white
    ax.tick_params(colors="white", which="both")

    # Set the y-axis limits based on RSI values
    ax.set_ylim(20, 80)

    ax.set_xlim(0, len(labels) + 2)

    # Remove the x-axis ticks since we're annotating each point
    ax.set_xticks([])

    add_legend(ax)

    plt.show()


def add_legend(ax):
    # Create custom legend handles with square markers, including BTC price
    adjusted_colors = list(COLORS_LABELS.values())
    # Change NEUTRAL color to grey
    adjusted_colors[2] = "#808080"
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color=BACKGROUND_COLOR,
            markerfacecolor=color,
            markersize=10,
            label=label,
        )
        for color, label in zip(
            adjusted_colors,
            [label.upper() for label in list(COLORS_LABELS.keys())],
        )
    ]

    # Add legend
    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(legend_handles),
        frameon=False,
        fontsize="small",
        labelcolor="white",
    )

    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight("bold")

    # Adjust layout to reduce empty space around the plot
    plt.subplots_adjust(left=0.05, right=0.95, top=0.875, bottom=0.1)


if __name__ == "__main__":
    plot_rsi_heatmap()
