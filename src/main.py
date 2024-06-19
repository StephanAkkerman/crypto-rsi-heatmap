import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI
from tradingview_ta import get_multiple_analysis

cg = CoinGeckoAPI()

FIGURE_SIZE = (20, 10)
BACKGROUND_COLOR = "#0d1117"
RANGES = {
    "Overbought": (70, 100),
    "Strong": (60, 70),
    "Neutral": (40, 60),
    "Weak": (30, 40),
    "Oversold": (0, 30),
}
COLORS_LABELS = {
    "Overbought": "#c32e3b",
    "Strong": "#681f28",
    "Neutral": "#0f2427",
    "Weak": "#144e48",
    "Oversold": "#1d8b7a",
}


def get_RSI(coins: list, exchange: str = "BINANCE") -> dict:
    # Format symbols exchange:symbol
    symbols = [f"{exchange.upper()}:{symbol}" for symbol in coins]

    analysis = get_multiple_analysis(symbols=symbols, interval="1d", screener="crypto")

    # For each symbol get the RSI
    rsi_dict = {}
    for symbol in symbols:
        if analysis[symbol] is None:
            # print(f"No analysis for {symbol}")
            continue
        clean_symbol = symbol.replace(f"{exchange.upper()}:", "")
        clean_symbol = clean_symbol.replace("USDT", "")
        rsi_dict[clean_symbol] = analysis[symbol].indicators["RSI"]

    return rsi_dict


def get_top_vol_coins(length: int = 100) -> list:
    CACHE_FILE = "data/top_vol_coins_cache.pkl"
    CACHE_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds
    # List of symbols to exclude
    STABLE_COINS = [
        "OKBUSDT",
        "DAIUSDT",
        "USDTUSDT",
        "USDCUSDT",
        "BUSDUSDT",
        "TUSDUSDT",
        "PAXUSDT",
        "EURUSDT",
        "GBPUSDT",
        "CETHUSDT",
        "WBTCUSDT",
    ]

    # Check if the cache file exists and is not expired
    os.makedirs(CACHE_FILE.split("/")[0], exist_ok=True)
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
            cache_time = cache_data["timestamp"]
            if time.time() - cache_time < CACHE_EXPIRATION:
                # Return the cached data if it's not expired
                print("Using cached top volume coins")
                return cache_data["data"][:length]

    # Fetch fresh data if the cache is missing or expired
    df = pd.DataFrame(cg.get_coins_markets("usd"))["symbol"].str.upper() + "USDT"

    sorted_volume = df[~df.isin(STABLE_COINS)]
    top_vol_coins = sorted_volume.tolist()

    # Save the result to the cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"timestamp": time.time(), "data": top_vol_coins}, f)

    return top_vol_coins[:length]


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
    legend_elements = []
    for start, end, color, label in color_map:
        ax.fill_between([0, len(labels) + 2], start, end, color=color, alpha=0.35)

        # Add the label to the custom legend
        if color == "black":
            color = "grey"
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=label))

    # Plot each point with a white border for visibility
    for i, label in enumerate(labels):
        ax.scatter(i + 1, rsi_values[i], color="white", edgecolor="black")
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
        len(labels) + 2,
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

    # Create the legend at the top
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        fancybox=True,
        shadow=True,
        frameon=False,
        fontsize="large",
        labelcolor="white",
    )

    # Display the plot
    plt.tight_layout()

    plt.show()


def add_legend(ax):
    # Create custom legend handles with square markers, including BTC price
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
            list(COLORS_LABELS.keys()), list(COLORS_LABELS.values())
        )
    ]

    # Add legend
    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
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
