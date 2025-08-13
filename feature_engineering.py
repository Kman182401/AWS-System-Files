import os
import boto3
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from io import BytesIO
import logging
logger = logging.getLogger(__name__)

# === Advanced Market Microstructure Features ===

def compute_order_flow_imbalance(order_book):
    """
    Calculate order flow imbalance from order book.
    order_book: dict with 'bids' and 'asks' (price, size)
    Returns: float (imbalance)
    """
    bid_volume = sum([size for price, size in order_book['bids']])
    ask_volume = sum([size for price, size in order_book['asks']])
    return (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)

def compute_queue_position(order_book, my_order_price, my_order_size, side='buy'):
    """
    Estimate queue position for a given order.
    """
    book_side = order_book['bids'] if side == 'buy' else order_book['asks']
    queue_ahead = sum([size for price, size in book_side if (price > my_order_price if side == 'buy' else price < my_order_price)])
    return queue_ahead

def compute_order_book_features(order_book):
    """
    Extracts order book analytics: spread, depth, top-of-book liquidity.
    """
    best_bid = order_book['bids'][0][0] if order_book['bids'] else None
    best_ask = order_book['asks'][0][0] if order_book['asks'] else None
    spread = best_ask - best_bid if best_bid and best_ask else None
    depth = sum([size for price, size in order_book['bids'] + order_book['asks']])
    return {
        'spread': spread,
        'depth': depth,
        'best_bid': best_bid,
        'best_ask': best_ask
    }

# S3 configuration (real values)
S3_BUCKET = "omega-singularity-ml"
RAW_DATA_PREFIX = ""  # Root of the bucket for raw data
PROCESSED_DATA_PREFIX = "processed"

# List of tickers to process
TICKERS = ["ES1!", "NQ1!", "GA", "GE", "GB", "GC"]

s3 = boto3.client("s3")

def list_s3_files(bucket, prefix):
    """List all files in S3 under the given prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def read_s3_file(bucket, key, data_type='TRADES'):
    """
    Read a file from S3 and return as a pandas DataFrame.
    Supports TRADES, MIDPOINT, etc. by modifying the key.
    """
    # Insert data_type into the filename before .csv or .parquet
    if key.endswith(".csv"):
        key = key.replace(".csv", f"_{data_type}.csv")
    elif key.endswith(".parquet"):
        key = key.replace(".parquet", f"_{data_type}.parquet")
    else:
        raise ValueError("Unsupported file type: " + key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    if key.endswith(".csv"):
        return pd.read_csv(obj["Body"])
    elif key.endswith(".parquet"):
        return pd.read_parquet(BytesIO(obj["Body"].read()))

def add_technical_indicators(df):
    """Add advanced technical indicators and features for ML/RL trading."""
    # Ensure required columns exist
    if not all(col in df.columns for col in ["close", "high", "low", "open", "volume"]):
        return df

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()  # Band width (volatility)

    # ATR (Average True Range)
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"]
    ).average_true_range()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"]
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # VWAP (Volume Weighted Average Price)
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    # Volume SMA and spikes
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
    df["vol_spike"] = (df["volume"] > 1.5 * df["vol_sma_20"]).astype(int)

    # Time-of-day/session encoding (minute of day, hour, day of week)
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
    # If no datetime column, just skip time-based features (no halt)

    # Lagged returns (1, 5, 15 periods)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    # Rolling correlation with open (as a proxy for related asset, can be replaced)
    df["roll_corr_open"] = df["close"].rolling(window=20).corr(df["open"])

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def process_ticker(ticker):
    """Process all files for a given ticker."""
    ticker_prefix = f"{RAW_DATA_PREFIX}{ticker}" if RAW_DATA_PREFIX else ticker
    processed_frames = []
    for key in list_s3_files(S3_BUCKET, ticker_prefix):
        df = read_s3_file(S3_BUCKET, key)
        df = add_technical_indicators(df)
        df["asset"] = ticker
        processed_frames.append(df)
    if processed_frames:
        return pd.concat(processed_frames, ignore_index=True)
    return None

def main():
    all_data = []
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df = process_ticker(ticker)
        if df is not None:
            all_data.append(df)
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Save to S3 as CSV
        output_key = f"{PROCESSED_DATA_PREFIX}/features_all_assets.csv"
        csv_buffer = BytesIO()
        final_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Processed features saved to s3://{S3_BUCKET}/{output_key}")
    else:
        print("No data processed.")

def generate_features(df):
    print("MARKET FEATURE ENGINEERING RUNNING")
    print("NEWS FEATURE ENGINEERING RUNNING")
    """
    Given a raw OHLCV DataFrame, add technical indicators and return features and binary up/down labels.
    Features: all columns except 'label'.
    Labels: 1 if next close > current close, else 0.
    """
    # Ensure required columns are present and correct type
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Data validation error: Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure timestamp column exists and is datetime
    if "timestamp" not in df.columns:
        if "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"])
        else:
            df["timestamp"] = pd.date_range(start="2000-01-01", periods=len(df), freq="min")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns + ["timestamp"])

    df = add_technical_indicators(df.copy())
    # Ensure a timestamp column exists for incremental learning
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    elif df.index.name is not None and np.issubdtype(df.index.dtype, np.datetime64):
        df["timestamp"] = df.index
    else:
        df["timestamp"] = np.arange(len(df))
    df = df.dropna()
    # Binary classification: 1 if next close > current close, else 0
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df.drop(['label'], axis=1).iloc[:-1]
    labels = df['label'].iloc[:-1]
    return features, labels

    print(df.columns)

def classify_news_type(headline, body=None):
    """
    Classifies news into types based on keywords.
    Args:
        headline (str): News headline.
        body (str, optional): News body/content.
    Returns:
        str: News type label ("earnings", "macro", "M&A", "regulatory", "other").
    """
    text = (headline or "") + " " + (body or "")
    text = text.lower()
    if any(word in text for word in ["earnings", "eps", "revenue", "profit", "quarterly results"]):
        return "earnings"
    if any(word in text for word in ["fed", "interest rate", "central bank", "inflation", "cpi", "macro", "gdp", "unemployment", "fomc", "ecb", "boe", "boj"]):
        return "macro"
    if any(word in text for word in ["merger", "acquisition", "buyout", "takeover", "m&a"]):
        return "M&A"
    if any(word in text for word in ["regulation", "regulatory", "sec", "fine", "settlement", "compliance", "lawsuit"]):
        return "regulatory"
    return "other"

def map_macro_news_to_tickers(text):
    """
    Maps macroeconomic or indirect news references to relevant tickers.
    Args:
        text (str): News headline or body.
    Returns:
        List[str]: List of tickers/entities inferred to be impacted.
    """
    text = text.lower()
    tickers = set()
    # Macro terms mapping
    macro_map = {
        "federal reserve": ["ES1!", "NQ1!", "USD", "GC"],
        "interest rate": ["ES1!", "NQ1!", "USD", "GC"],
        "inflation": ["ES1!", "NQ1!", "USD", "GC"],
        "cpi": ["ES1!", "NQ1!", "USD", "GC"],
        "gdp": ["ES1!", "NQ1!", "USD", "GC"],
        "ecb": ["GE"],
        "boe": ["GB"],
        "rba": ["GA"],
        "trade deal": ["ES1!", "NQ1!", "USD", "GC"],
        "oil": ["GC"],
    }
    for macro_term, macro_tickers in macro_map.items():
        if macro_term in text:
            tickers.update(macro_tickers)
    return list(tickers)

def extract_features_from_level2_market_data(market_data, depth=10):
    """
    Extract advanced features from Level 2 (order book) market data.
    Assumes market_data contains 'bids' and 'asks' as lists of [price, size] pairs.
    """
    features = {}

    # Extract top-of-book (Level 1) features
    if market_data.get("bids") and market_data.get("asks"):
        best_bid = market_data["bids"][0][0]
        best_ask = market_data["asks"][0][0]
        features["best_bid"] = best_bid
        features["best_ask"] = best_ask
        features["spread"] = best_ask - best_bid
        features["bid_size"] = market_data["bids"][0][1]
        features["ask_size"] = market_data["asks"][0][1]
    else:
        features["best_bid"] = 0
        features["best_ask"] = 0
        features["spread"] = 0
        features["bid_size"] = 0
        features["ask_size"] = 0

    # Level 2 features: order book depth, imbalance, etc.
    depth = min(len(market_data.get("bids", [])), len(market_data.get("asks", [])), depth)
    features["book_imbalance"] = (
        sum([b[1] for b in market_data.get("bids", [])[:depth]]) -
        sum([a[1] for a in market_data.get("asks", [])[:depth]])
    )
    features["bid_depth"] = sum([b[1] for b in market_data.get("bids", [])[:depth]])
    features["ask_depth"] = sum([a[1] for a in market_data.get("asks", [])[:depth]])

    # Optionally, add more advanced features here

    return features

if __name__ == "__main__":
    main()
