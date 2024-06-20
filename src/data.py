import os
import pickle
import time

import pandas as pd
from pycoingecko import CoinGeckoAPI
from tradingview_ta import get_multiple_analysis

cg = CoinGeckoAPI()


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
