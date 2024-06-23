import os
import pickle
import time
from datetime import datetime, timedelta

import pandas as pd
from pycoingecko import CoinGeckoAPI
from tradingview_ta import get_multiple_analysis

cg = CoinGeckoAPI()


def get_RSI(coins: list, exchange: str = "BINANCE", time_frame: str = "1d") -> dict:
    # Format symbols exchange:symbol
    symbols = [f"{exchange.upper()}:{symbol}" for symbol in coins]

    analysis = get_multiple_analysis(
        symbols=symbols, interval=time_frame, screener="crypto"
    )

    # For each symbol get the RSI
    rsi_dict = {}
    for symbol in symbols:
        if analysis[symbol] is None:
            # print(f"No analysis for {symbol}")
            continue
        clean_symbol = symbol.replace(f"{exchange.upper()}:", "")
        clean_symbol = clean_symbol.replace("USDT", "")
        rsi_dict[clean_symbol] = analysis[symbol].indicators["RSI"]

    # Save the RSI data to a CSV file
    save_RSI(rsi_dict, time_frame)

    return rsi_dict


def get_closest_to_24h(
    file_path: str = "data/rsi_data.csv", time_frame: str = "1d"
) -> dict:
    # Read the CSV file into a DataFrame
    if not os.path.isfile(file_path):
        print(f"No data found in {file_path}")
        return {}

    df = pd.read_csv(file_path)

    # Filter on the timeframe
    df = df[df["Time Frame"] == time_frame]

    # Convert the 'Date' column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Calculate the time difference from 24 hours ago
    target_time = datetime.now() - timedelta(hours=24)
    df["Time_Diff"] = abs(df["Date"] - target_time)

    # Find the minimum time difference
    min_time_diff = df["Time_Diff"].min()

    # Filter rows that have the minimum time difference
    closest_rows = df[df["Time_Diff"] == min_time_diff]

    # Convert the filtered rows to a dictionary with symbols as keys and RSI as values
    result = closest_rows.set_index("Symbol")["RSI"].to_dict()

    return result


def save_RSI(
    rsi_dict: dict, time_frame: str, file_path: str = "data/rsi_data.csv"
) -> None:
    # Convert the RSI dictionary to a DataFrame
    df = pd.DataFrame(list(rsi_dict.items()), columns=["Symbol", "RSI"])

    # Add the current date to the DataFrame
    df["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["Time Frame"] = time_frame

    # Check if the file exists
    if os.path.isfile(file_path):
        # Append data to the existing CSV file
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        # Save the DataFrame to a new CSV file with header
        df.to_csv(file_path, index=False)

    print(f"RSI data saved to {file_path}")


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
