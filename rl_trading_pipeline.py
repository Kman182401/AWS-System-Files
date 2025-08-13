try:
    from market_data_utils import RealTimeFeatureStore
except Exception:
    class RealTimeFeatureStore:
        def __init__(self):
            self._store = {}
        def update(self, key, features_dict):
            import pandas as pd
            # Store as a growing DataFrame; coerce to numeric where possible
            df = self._store.get(key)
            row = pd.DataFrame([features_dict])
            self._store[key] = row if df is None else pd.concat([df, row], ignore_index=True).infer_objects(copy=False)
        def get(self, key):
            return self._store.get(key)
        def clear(self):
            self._store.clear()
feature_store = RealTimeFeatureStore()

from market_data_config import MAX_POSITION_EXPOSURE
import datetime
from market_data_config import MAX_DAILY_LOSS_PCT, MAX_TRADES_PER_DAY
from ib_insync import MarketOrder
from market_data_ibkr_adapter import IBKRIngestor
from market_data_config import IBKR_SYMBOLS
TICKERS = IBKR_SYMBOLS
import psutil
from feature_engineering import classify_news_type
from sklearn.linear_model import LogisticRegression
import joblib
import os
import numpy as np
import random
import math
import json
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import boto3
import io
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict
from feature_engineering import generate_features, add_technical_indicators, process_ticker, list_s3_files, read_s3_file
try:
    from ingest_to_s3 import upload_file_to_s3
except Exception:
    def upload_file_to_s3(local_path: str, bucket: str, key: str):
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)

from market_data_ibkr_adapter import IBKRIngestor

from feature_engineering import generate_features
from audit_logging_utils import log_trade_results

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import pandas as pd  # <-- Ensure pandas is imported for mock data
import logging
from colorlog import ColoredFormatter

import warnings
from stable_baselines3.common.callbacks import BaseCallback

import logging
logger = logging.getLogger(__name__)

LAST_TRAINED_FILE = "last_trained_time.txt"

from news_ingestion_marketaux import fetch_marketaux_news, normalize_marketaux_news
from news_data_utils import engineer_news_features

from market_data_config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_SYMBOLS, IBKR_LOG_LEVEL,
    IBKR_RETRY_INTERVAL, IBKR_TIMEOUT, IBKR_MARKET_DATA_TYPE,
    IBKR_LOGGING_ENABLED, IBKR_LOG_FILE, IBKR_DYNAMIC_SUBSCRIPTION,
    IBKR_RESOURCE_MONITORING, IBKR_POLL_INTERVAL
)

RETRAIN_INTERVAL = 24 * 3600  # 24 hours in seconds
last_retrain_time = 0

def detect_regime(price_series, window=100):
    returns = price_series.pct_change().dropna()
    rolling_vol = returns.rolling(window).std()
    high_vol = rolling_vol > rolling_vol.median() * 1.5
    if high_vol.iloc[-1]:
        return "high_vol"
    else:
        return "normal"

def new_data_available(data_dir, ticker, last_check_time):
    filename = f"{ticker}_TRADES.csv"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    return mtime > last_check_time

def execute_trade(ib, contract, action, quantity):
    """
    Place a market order via IBKR API.
    ib: ib_insync.IB instance (connected)
    contract: ib_insync.Contract instance (for the symbol)
    action: 'BUY' or 'SELL'
    quantity: int
    """
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    print(f"Order submitted: {action} {quantity} {contract.localSymbol}")
    return trade

IBKR_CONFIG = {
    'host': IBKR_HOST,
    'port': IBKR_PORT,
    'client_id': IBKR_CLIENT_ID,
    'symbols': IBKR_SYMBOLS,
    'log_level': IBKR_LOG_LEVEL,
    'retry_interval': IBKR_RETRY_INTERVAL,
    'timeout': IBKR_TIMEOUT,
    'market_data_type': IBKR_MARKET_DATA_TYPE,
    'logging_enabled': IBKR_LOGGING_ENABLED,
    'log_file': IBKR_LOG_FILE,
    'dynamic_subscription': IBKR_DYNAMIC_SUBSCRIPTION,
    'resource_monitoring': IBKR_RESOURCE_MONITORING,
    'poll_interval': IBKR_POLL_INTERVAL
}

bucket = "omega-singularity-ml"

# MarketAux entity types: include indices, commodities, currencies, and futures if supported
entity_types = "indices,commodities,currencies,futures"

# Major forex countries based on your pairs: US, GB, EU, AU
countries = "US,GB,EU,AU"

# Fetch latest news from MarketAux, filtered for these tickers/entities and types
news_list = fetch_marketaux_news(
    symbols=TICKERS,
    limit=20,  # Adjust as needed
    entity_types=entity_types,
    countries="us",
    language="en",
    filter_entities="false",
    must_have_entities="false",
    min_match_score=0.05
)
news_df = normalize_marketaux_news(news_list)
print(news_df.head(0))  # For verification
if news_df.empty:
    print("No news data found. Skipping analysis for this run.")
    # Prevent further processing on empty DataFrame
    exit()  # or return if inside a function
    # Optionally: return or continue, depending on your pipeline structure
    # If this is in main(), use: return
    # If this is in a loop, use: continue
    # If this is top-level, you can just let the script end naturally
from feature_engineering import map_macro_news_to_tickers

# Add a column for mapped macro tickers
# Use 'description' if available, otherwise 'title'
if 'description' in news_df.columns:
    news_df['macro_mapped_tickers'] = news_df['description'].apply(map_macro_news_to_tickers)
elif 'title' in news_df.columns:
    news_df['macro_mapped_tickers'] = news_df['title'].apply(map_macro_news_to_tickers)
else:
    print("No suitable text column found for macro mapping.")
    news_df['macro_mapped_tickers'] = [[] for _ in range(len(news_df))]

# Optionally, merge with direct tickers/entities if present
def merge_ticker_lists(row):
    direct = set(row.get('tickers', []))
    macro = set(row.get('macro_mapped_tickers', []))
    return list(direct.union(macro))

news_df['all_relevant_tickers'] = news_df.apply(merge_ticker_lists, axis=1)

from news_data_utils import compute_news_embeddings, find_similar_news

# Step 1: Compute embeddings for all news articles
embeddings = compute_news_embeddings(news_df)

# Step 2: For now, use news_df as a placeholder for historical impactful news
# (Replace with your actual impactful news DataFrame when available)
historical_impactful_news_df = news_df
historical_embeddings = compute_news_embeddings(historical_impactful_news_df)

from news_data_utils import compute_news_embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Compute embeddings for new and historical news
new_embeddings = compute_news_embeddings(news_df, text_col='description')
historical_embeddings = compute_news_embeddings(historical_impactful_news_df, text_col='description')

# Compute similarity matrix
similarity_matrix = cosine_similarity(new_embeddings, historical_embeddings)

top_k = 1
top_indices = similarity_matrix.argsort(axis=1)[:, -top_k:][:, ::-1]

news_df['most_similar_impactful_news'] = [
    [historical_impactful_news_df.iloc[idx]['description'] for idx in indices]
    for indices in top_indices
]
news_df['similarity_score'] = [
    [similarity_matrix[i, idx] for idx in indices]
    for i, indices in enumerate(top_indices)
]

# Now safe to proceed
embeddings = compute_news_embeddings(news_df)

es1_price_df = df_trades.merge(df_midpoint, on="timestamp", suffixes=('_trades', '_midpoint'))
print("="*40)
print("Price Impact Analysis for ES1! (Significance Detection):")
if news_df.empty or es1_price_df.empty:
    print("  No news or price data available for ES1!")
else:
    for idx, news_row in news_df.iterrows():
        event_time = pd.to_datetime(news_row['published_at'])
        # Ensure es1_price_df index is timezone-aware (UTC)
    if es1_price_df.index.tz is None:
        es1_price_df.index = es1_price_df.index.tz_localize('UTC')
    # Ensure event_time is timezone-aware (UTC)
    if getattr(event_time, 'tzinfo', None) is None or event_time.tzinfo is None:
        event_time = event_time.tz_localize('UTC')
    else:
        event_time = event_time.tz_convert('UTC')
    if event_time not in es1_price_df.index:
        # Find the nearest index using get_indexer
        nearest_idx = es1_price_df.index.get_indexer([event_time], method='nearest')[0]
        event_time = es1_price_df.index[nearest_idx]
        result = is_price_move_significant(
            price_series=es1_price_df['close'],
            event_time=event_time,
            lookback=60,
            post_event=5,
            z_thresh=2.0
        )
        if result['is_significant']:
            print(f"  Significant move after news at {event_time}: z={result['z_score']:.2f}, move={result['price_move']:.2%}, summary: {news_row['summary']}")
        else:
            print(f"  No significant move after news at {event_time}.")
print("="*40)

nq1_price_df = df_trades.merge(df_midpoint, on="timestamp", suffixes=('_trades', '_midpoint'))
print("="*40)
print("Price Impact Analysis for NQ1! (Significance Detection):")
if news_df.empty or nq1_price_df.empty:
    print("  No news or price data available for NQ1!")
else:
    for idx, news_row in news_df.iterrows():
        event_time = pd.to_datetime(news_row['published_at'])
    # Ensure nq1_price_df index is timezone-aware (UTC)
    if nq1_price_df.index.tz is None:
        nq1_price_df.index = nq1_price_df.index.tz_localize('UTC')
    # Ensure event_time is timezone-aware (UTC)
    if getattr(event_time, 'tzinfo', None) is None or event_time.tzinfo is None:
        event_time = event_time.tz_localize('UTC')
    else:
        event_time = event_time.tz_convert('UTC')
    if event_time not in nq1_price_df.index:
        # Find the nearest index using get_indexer
        nearest_idx = nq1_price_df.index.get_indexer([event_time], method='nearest')[0]
        event_time = nq1_price_df.index[nearest_idx]
        result = is_price_move_significant(
            price_series=nq1_price_df['close'],
            event_time=event_time,
            lookback=60,
            post_event=5,
            z_thresh=2.0
        )
        if result['is_significant']:
            print(f"  Significant move after news at {event_time}: z={result['z_score']:.2f}, move={result['price_move']:.2%}, summary: {news_row['summary']}")
        else:
            print(f"  No significant move after news at {event_time}.")
print("="*40)

gbpusd_price_df = df_trades.merge(df_midpoint, on="timestamp", suffixes=('_trades', '_midpoint'))
print("="*40)
print("Price Impact Analysis for GBPUSD (Significance Detection):")
if news_df.empty or gbpusd_price_df.empty:
    print("  No news or price data available for GBPUSD")
else:
    for idx, news_row in news_df.iterrows():
        event_time = pd.to_datetime(news_row['published_at'])
    # Ensure gbpusd_price_df index is timezone-aware (UTC)
    if gbpusd_price_df.index.tz is None:
        gbpusd_price_df.index = gbpusd_price_df.index.tz_localize('UTC')
    # Ensure event_time is timezone-aware (UTC)
    if getattr(event_time, 'tzinfo', None) is None or event_time.tzinfo is None:
        event_time = event_time.tz_localize('UTC')
    else:
        event_time = event_time.tz_convert('UTC')
    if event_time not in gbpusd_price_df.index:
        # Find the nearest index using get_indexer
        nearest_idx = gbpusd_price_df.index.get_indexer([event_time], method='nearest')[0]
        event_time = gbpusd_price_df.index[nearest_idx]
        result = is_price_move_significant(
            price_series=gbpusd_price_df['close'],
            event_time=event_time,
            lookback=60,
            post_event=5,
            z_thresh=2.0
        )
        if result['is_significant']:
            print(f"  Significant move after news at {event_time}: z={result['z_score']:.2f}, move={result['price_move']:.2%}, summary: {news_row['summary']}")
        else:
            print(f"  No significant move after news at {event_time}.")
print("="*40)

eurusd_price_df = df_trades.merge(df_midpoint, on="timestamp", suffixes=('_trades', '_midpoint'))
print("="*40)
print("Price Impact Analysis for EURUSD (Significance Detection):")
if news_df.empty or eurusd_price_df.empty:
    print("  No news or price data available for EURUSD")
else:
    for idx, news_row in news_df.iterrows():
        event_time = pd.to_datetime(news_row['published_at'])
    # Ensure eurusd_price_df index is timezone-aware (UTC)
    if eurusd_price_df.index.tz is None:
        eurusd_price_df.index = eurusd_price_df.index.tz_localize('UTC')
    # Ensure event_time is timezone-aware (UTC)
    if getattr(event_time, 'tzinfo', None) is None or event_time.tzinfo is None:
        event_time = event_time.tz_localize('UTC')
    else:
        event_time = event_time.tz_convert('UTC')
    if event_time not in eurusd_price_df.index:
        # Find the nearest index using get_indexer
        nearest_idx = eurusd_price_df.index.get_indexer([event_time], method='nearest')[0]
        event_time = eurusd_price_df.index[nearest_idx]
        result = is_price_move_significant(
            price_series=eurusd_price_df['close'],
            event_time=event_time,
            lookback=60,
            post_event=5,
            z_thresh=2.0
        )
        if result['is_significant']:
            print(f"  Significant move after news at {event_time}: z={result['z_score']:.2f}, move={result['price_move']:.2%}, summary: {news_row['summary']}")
        else:
            print(f"  No significant move after news at {event_time}.")
print("="*40)

audusd_price_df = df_trades.merge(df_midpoint, on="timestamp", suffixes=('_trades', '_midpoint'))
print("="*40)
print("Price Impact Analysis for AUDUSD (Significance Detection):")
if news_df.empty or audusd_price_df.empty:
    print("  No news or price data available for AUDUSD")
else:
    for idx, news_row in news_df.iterrows():
        event_time = pd.to_datetime(news_row['published_at'])
    # Ensure audusd_price_df index is timezone-aware (UTC)
    if audusd_price_df.index.tz is None:
        audusd_price_df.index = audusd_price_df.index.tz_localize('UTC')
    # Ensure event_time is timezone-aware (UTC)
    if getattr(event_time, 'tzinfo', None) is None or event_time.tzinfo is None:
        event_time = event_time.tz_localize('UTC')
    else:
        event_time = event_time.tz_convert('UTC')
    if event_time not in audusd_price_df.index:
        # Find the nearest index using get_indexer
        nearest_idx = audusd_price_df.index.get_indexer([event_time], method='nearest')[0]
        event_time = audusd_price_df.index[nearest_idx]
        result = is_price_move_significant(
            price_series=audusd_price_df['close'],
            event_time=event_time,
            lookback=60,
            post_event=5,
            z_thresh=2.0
        )
        if result['is_significant']:
            print(f"  Significant move after news at {event_time}: z={result['z_score']:.2f}, move={result['price_move']:.2%}, summary: {news_row['summary']}")
        else:
            print(f"  No significant move after news at {event_time}.")
print("="*40)

gold_price_df = df_trades.merge(df_midpoint, on="timestamp", suffixes=('_trades', '_midpoint'))
print("="*40)
print("Price Impact Analysis for XAUUSD (Significance Detection):")
if news_df.empty or xauusd_price_df.empty:
    print("  No news or price data available for XAUUSD")
else:
    for idx, news_row in news_df.iterrows():
        event_time = pd.to_datetime(news_row['published_at'])
    # Ensure xauusd_price_df index is timezone-aware (UTC)
    if xauusd_price_df.index.tz is None:
        xauusd_price_df.index = xauusd_price_df.index.tz_localize('UTC')
    # Ensure event_time is timezone-aware (UTC)
    if getattr(event_time, 'tzinfo', None) is None or event_time.tzinfo is None:
        event_time = event_time.tz_localize('UTC')
    else:
        event_time = event_time.tz_convert('UTC')
    if event_time not in xauusd_price_df.index:
        # Find the nearest index using get_indexer
        nearest_idx = xauusd_price_df.index.get_indexer([event_time], method='nearest')[0]
        event_time = xauusd_price_df.index[nearest_idx]
        result = is_price_move_significant(
            price_series=xauusd_price_df['close'],
            event_time=event_time,
            lookback=60,
            post_event=5,
            z_thresh=2.0
        )
        if result['is_significant']:
            print(f"  Significant move after news at {event_time}: z={result['z_score']:.2f}, move={result['price_move']:.2%}, summary: {news_row['summary']}")
        else:
            print(f"  No significant move after news at {event_time}.")
print("="*40)

class PPOTrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_history = []

    def _on_step(self) -> bool:
        logs = self.locals.get('logs', None)
        if logs and 'rollout/ep_rew_mean' in logs:
            self.log_history.append({
                "Step": logs.get("total_timesteps", 0),
                "Avg. Ep. Length": logs.get("rollout/ep_len_mean", 0),
                "Avg. Ep. Reward": logs.get("rollout/ep_rew_mean", 0),
                "Explained Variance": logs.get("train/explained_variance", 0),
                "Loss": logs.get("train/loss", 0),
                "Value Loss": logs.get("train/value_loss", 0),
                "Policy Loss": logs.get("train/policy_gradient_loss", 0),
            })
        return True

    def print_summary(self):
        if not self.log_history:
            print("No PPO training logs collected.")
            return
        df = pd.DataFrame(self.log_history)
        print("\n=== PPO RL Training Progress Summary ===")
        print(df.to_string(index=False))
        print("\nKey:")
        print("  Step: Total environment steps so far")
        print("  Avg. Ep. Length: Average steps per episode")
        print("  Avg. Ep. Reward: Average total reward per episode (higher is better)")
        print("  Explained Variance: Value function fit quality (1=perfect, 0=random, <0=bad)")
        print("  Loss: Total loss (lower is better)")
        print("  Value Loss: Value function loss")
        print("  Policy Loss: Policy gradient loss")

LOG_LEVEL = logging.INFO  # Change to DEBUG for more detail, WARNING for less

LOG_FORMAT = (
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s"
    "\n" + "-"*80
)

LOG_COLORS = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'bold_red',
}

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", log_colors=LOG_COLORS))

root_logger = logging.getLogger()
root_logger.handlers = []  # Remove any existing handlers
root_logger.addHandler(handler)
root_logger.setLevel(LOG_LEVEL)
# Only show warnings/errors by default
logging.getLogger().setLevel(logging.WARNING)

def to_scalar(x):
    """Convert any array-like or scalar to a Python int."""
    arr = np.asarray(x)
    if arr.shape == ():  # scalar
        return int(arr.item())
    elif arr.shape == (1,):  # 1-element array
        return int(arr[0])
    else:
        logger.error(f"Non-scalar value encountered: {x} (type: {type(x)}, shape: {arr.shape})")
        raise ValueError(f"Non-scalar value encountered: {x} (type: {type(x)}, shape: {arr.shape})")

def save_model_to_s3(model, bucket, key):
    """Save a model to S3 using joblib."""
    local_path = f"/tmp/{os.path.basename(key)}"
    joblib.dump(model, local_path)
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    os.remove(local_path)

def load_model_from_s3(bucket, key):
    """Load a model from S3 using joblib."""
    local_path = f"/tmp/{os.path.basename(key)}"
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    model = joblib.load(local_path)
    os.remove(local_path)
    return model

def get_latest_model_key(bucket, prefix, exclude_run_id=None):
    """Get the latest model key from S3, optionally excluding the current run_id."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [obj['Key'] for obj in response.get('Contents', [])]
    if exclude_run_id:
        keys = [k for k in keys if exclude_run_id not in k]
    if not keys:
        return None
    return sorted(keys)[-1]  # Latest by name (run_id)

# === Add this drift detection function here ===
def detect_feature_drift(reference_data, new_data, threshold=3.0):
    drift_detected = False
    for col in reference_data.columns:
        ref_mean, ref_std = reference_data[col].mean(), reference_data[col].std()
        new_mean = new_data[col].mean()
        if abs(ref_mean - new_mean) > threshold * ref_std:
            drift_detected = True
            logging.warning(f"Drift detected in {col}: ref_mean={ref_mean}, new_mean={new_mean}")
    return drift_detected
# === End drift detection function ===

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.transaction_cost = 0.0005  # 5 bps per trade, adjust as needed
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        self.risk_penalty_coeff = 0.1  # adjust for risk aversion
        self.action_space = gym.spaces.Discrete(3)  # [0: hold, 1: buy, 2: sell]
        # Drop datetime columns from data for observation space
        self.numeric_columns = [col for col in self.data.columns if not np.issubdtype(self.data[col].dtype, np.datetime64)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.numeric_columns),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        if seed is not None:
            np.random.seed(seed)
        obs = self.data.iloc[self.current_step][self.numeric_columns].values.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        reward = 0
        done = False
        prev_position = self.position
        prev_balance = self.balance

        # Transaction cost if position changes
        trade_cost = 0
        if action == 1:  # buy
            if self.position != 1:
                trade_cost = self.transaction_cost * self.data.iloc[self.current_step]['close']
            self.position = 1
        elif action == 2:  # sell
            if self.position != -1:
                trade_cost = self.transaction_cost * self.data.iloc[self.current_step]['close']
            self.position = -1
        else:  # hold
            self.position = 0

        # Reward: PnL from position minus transaction cost
        if self.current_step > 0:
            price_diff = self.data.iloc[self.current_step]['close'] - self.data.iloc[self.current_step - 1]['close']
            reward = prev_position * price_diff - trade_cost

        # Update balance and drawdown
        self.balance += reward
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Risk penalty (drawdown)
        risk_penalty = self.risk_penalty_coeff * self.max_drawdown
        reward -= risk_penalty

        # Risk limit enforcement
        if self.max_drawdown > 0.2:  # 20% drawdown hard stop
            done = True
            logging.warning("Max drawdown limit reached. Ending episode.")

        self.current_step += 1
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            done = True

        # Prepare next observation
        if self.current_step >= len(self.data):
            obs = np.zeros(len(self.numeric_columns), dtype=np.float32)
        else:
            obs = self.data.iloc[self.current_step][self.numeric_columns].values.astype(np.float32)

        terminated = done  # True if episode ended due to terminal state (e.g., max drawdown, end of data)
        truncated = False  # True if episode ended due to time/trade limit (set to False if not used)
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

import re
import joblib
import boto3
import os
from stable_baselines3 import PPO

def load_all_models(bucket, run_id):
    """
    Loads all models (ML and PPO) for a given run_id from S3.
    Returns a nested dict: {ticker: {model_name: model_object}}
    """
    s3 = boto3.client("s3")
    prefix = f"models/{run_id}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    models = {}
    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = os.path.basename(key)
        # ML models: {ticker}_{modelname}.joblib
        m = re.match(r"(.+?)_(RandomForest|XGBoost|LightGBM)\.joblib$", filename)
        if m:
            ticker, model_name = m.groups()
            local_path = f"/tmp/{filename}"
            s3.download_file(bucket, key, local_path)
            model = joblib.load(local_path)
            os.remove(local_path)
            if ticker not in models:
                models[ticker] = {}
            models[ticker][model_name] = model
        # PPO RL models: {ticker}_ppo.zip
        m2 = re.match(r"(.+?)_ppo\.zip$", filename)
        if m2:
            ticker = m2.group(1)
            local_path = f"/tmp/{filename}"
            s3.download_file(bucket, key, local_path)
            model = PPO.load(local_path)
            os.remove(local_path)
            if ticker not in models:
                models[ticker] = {}
            models[ticker]["PPO"] = model
    return models

class RLTradingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.market_data_adapter = IBKRIngestor()
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 10)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.2)  # 20% default
        self.max_position_size = config.get('max_position_size', 1)      # 1 contract/lot default
        self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.status = "initialized"
        self.last_error = None
        self.notification_hook = None  # Extensibility: notification system

    def run(self):
        self.status = "running"
        log_trade_results(f"pipeline_start: run_id={self.run_id}")
        try:
            self._main_loop()
            self.status = "completed"
            log_trade_results(f"pipeline_complete: run_id={self.run_id}")
        except Exception as e:
            self.status = "failed"
            self.last_error = str(e)
            log_trade_results(f"pipeline_error: run_id={self.run_id}, error={self.last_error}, traceback={traceback.format_exc()}")
            logger.error(f"Pipeline failed: {e}")
            self.notify(f"Pipeline failed: {e}")
            raise

    def _main_loop(self):
        retries = 0
        while retries < self.max_retries:
            try:
                logging.info("Fetching market data...")
                # List of tickers you want to fetch (using your mapping for yfinance)
                tickers = ["ES1!", "NQ1!", "GA", "GB", "GE", "GC"]
                last_check_times = {ticker: 0 for ticker in tickers}
                raw_data = {}
                for idx, ticker in enumerate(tickers):
                    start_time = time.time()
                    try:
                        if self.config.get("use_mock_data", True):
                            df = generate_mock_market_data(length=500, seed=42 + tickers.index(ticker))
                            logging.info(f"{ticker} mock data shape: {df.shape}")
                        else:
                            df = self.market_data_adapter.fetch_data(ticker)
                            logging.info(f"{ticker} raw data shape: {df.shape}")
                            if df is None or df.empty:
                                logging.warning(f"No data for {ticker}, skipping.")
                                continue
                            raw_data[ticker] = df
                            logging.info(f"Market data fetched successfully.")
                            print(f"{ticker} columns: {df.columns}")

                            # --- Regime Detection and Alerting ---
                            regime = detect_regime(df['close'])
                            if regime == "high_vol":
                                send_sms_alert(f"High volatility regime detected for {ticker}!")
                                logging.info(f"Regime change: High volatility detected for {ticker}")
                                print(f"High volatility regime detected for {ticker}, triggering retraining/model switch.")
                                # Trigger retraining/model switch
                                regime = detect_regime(df['close'])
                                if regime == "high_vol":
                                    model_path = f"/home/ubuntu/models/ppo_model_{ticker}_highvol.zip"
                                else:
                                    model_path = f"/home/ubuntu/models/ppo_model_{ticker}_normal.zip"
                                self.train_ppo(df, ticker, save_path=model_path)
                                last_check_times[ticker] = time.time()
                                # Optionally, you could load a different model here if you have one for high-vol regimes

                        # --- PPO Training for Each Ticker Sequentially ---
                        current_time = time.time()
                        if (current_time - last_retrain_time) > RETRAIN_INTERVAL or new_data_available(data_dir, ticker, last_check_times[ticker]):
                            print(f"Training PPO for {ticker} (scheduled or new data)...")
                            regime = detect_regime(df['close'])
                            if regime == "high_vol":
                                model_path = f"/home/ubuntu/models/ppo_model_{ticker}_highvol.zip"
                            else:
                                model_path = f"/home/ubuntu/models/ppo_model_{ticker}_normal.zip"
                            self.train_ppo(df, ticker, save_path=model_path)
                            last_check_times[ticker] = current_time
                            last_retrain_time = current_time
                            # (rest of your retraining logic)
                        else:
                            print(f"No new data or scheduled retrain for {ticker}, skipping retraining.")

                        # Log resource usage after PPO training
                        cpu = psutil.cpu_percent()
                        mem = psutil.virtual_memory().percent
                        print(f"Resource usage after {ticker}: CPU {cpu}%, RAM {mem}%")

                        # Progress logging
                        elapsed = time.time() - start_time
                        print(f"Finished PPO for {ticker} ({idx+1}/{len(tickers)}) in {elapsed:.1f} seconds.")

                        # Dynamic sleep based on resource usage
                        if cpu > 80 or mem > 80:
                            print("High resource usage detected, sleeping for 30 seconds...")
                            time.sleep(30)
                        else:
                            time.sleep(10)  # Default sleep

                    except Exception as e:
                        print(f"ERROR during PPO training for {ticker}: {e}")
                        traceback.print_exc()
                        print(f"Skipping {ticker} and continuing to next ticker...")

                print("All tickers processed. Check above for any errors or resource warnings.")

                # --- ADD THIS BLOCK IMMEDIATELY AFTER FETCHING DATA, BEFORE FEATURE ENGINEERING ---
                # (This should NOT be indented; it should be at the same level as the print above)
                # If you want to generate features for all tickers after the PPO loop, do it here:
                # Example: for all tickers in raw_data
                for ticker, df in raw_data.items():
                    features, labels = generate_features(df)
                    logger.info(f"Ticker [{ticker}]: features shape = {features.shape}, labels shape = {labels.shape}")

                logging.info("Engineering features...")

                # Load last trained timestamp/index
                try:
                    with open(LAST_TRAINED_FILE, "r") as f:
                        last_trained_time = f.read().strip()
                        if last_trained_time:
                            # Convert to int (or pd.to_datetime if using datetime)
                            last_trained_time = int(last_trained_time)
                except FileNotFoundError:
                    last_trained_time = None

                features = {}
                labels = {}

                for ticker, df in raw_data.items():
                    X, y = generate_features(df)
                    if last_trained_time and "timestamp" in X.columns:
                        if np.issubdtype(X["timestamp"].dtype, np.datetime64):
                            last_trained_time = pd.to_datetime(last_trained_time)
                        else:
                            last_trained_time = int(last_trained_time)
                        X = X[X["timestamp"] > last_trained_time]
                        y = y.loc[X.index]
                    logging.info(f"{ticker} features shape: {X.shape}, labels shape: {y.shape}")
                    if X is None or X.empty or y is None or y.empty:
                        logging.warning(f"No features/labels for {ticker} after feature engineering, skipping.")
                        continue
                    features[ticker] = X
                    labels[ticker] = y
                logging.info("Features engineered successfully.")

                # === DRIFT DETECTION BLOCK ===
                # AI_ASSISTANT_TODO: CRITICAL REMINDER!
                # If you are enabling live IBKR data, you MUST set 'disable_drift_detection' to False in the config below.
                # This is a production safeguard. If you are an AI assistant or developer, PROMPT THE USER to re-enable drift detection
                # before allowing any live trading or IBKR integration. This is essential for risk management and compliance.
                for ticker, X in features.items():
                    if len(X) > 200:
                        if not self.config.get("disable_drift_detection", False):
                            reference = X.iloc[:100]
                            new = X.iloc[-100:]
                            if detect_feature_drift(reference, new, threshold=self.config.get("drift_detection_threshold", 3.0)):
                                logger.error(f"Feature drift detected for {ticker}. Halting pipeline.")
                                self.status = "failed"
                                self.notify(f"Feature drift detected for {ticker}.")
                                # Automated model rollback logic
                                logging.warning("Rolling back to previous model version due to error/drift.")
                                self.notify("Model rollback triggered.")
                                return
                        else:
                            logging.warning(
                                "Drift detection is DISABLED for testing. "
                                "AI_ASSISTANT_TODO: When enabling live IBKR data, RE-ENABLE drift detection in config!"
                            )

                # --- ML Model Training and Prediction Block ---
                from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
                from sklearn.ensemble import RandomForestClassifier
                logging.warning("ML SYSTEM: Running ML model training and prediction for all tickers.")
                from sklearn.metrics import accuracy_score, confusion_matrix

                # Define models
                models = {
                    "SGDClassifier": SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-3),
                    "Perceptron": Perceptron(),
                    "PassiveAggressive": PassiveAggressiveClassifier(),
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)  # Optional batch model
                }

                fitted_models = {}

                # Metrics storage
                results = {}

                for ticker in features:
                    X = feature_store.get(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    # Optional: Log the features being used
                    logging.info(f"Features for {ticker}: {X}")

                    if len(y) == 0:
                        logging.warning(f"No labels for {ticker}, skipping ML for this ticker.")
                        continue

                    # Split into train/test (last 20% for test)
                    split_idx = int(len(X) * 0.95)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                    # Drop any datetime or identifier columns from features
                    cols_to_drop = []
                    for col in X_train.columns:
                        if np.issubdtype(X_train[col].dtype, np.datetime64) or col in ["datetime", "timestamp"]:
                            cols_to_drop.append(col)
                    if cols_to_drop:
                        X_train = X_train.drop(columns=cols_to_drop)
                        X_test = X_test.drop(columns=cols_to_drop)

                    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                        logging.warning(f"Not enough data for {ticker}: X_train={len(X_train)}, X_test={len(X_test)}. Skipping.")
                        continue

                    ticker_results = {}

                    for name, model in models.items():
                        model_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/{ticker}_{name}")
                        if model_key:
                            prev_model = load_model_from_s3(self.config['s3_bucket'], model_key)
                            if hasattr(prev_model, "partial_fit"):
                                if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                                    logger.error("No data available for model training.")
                                    raise ValueError("No data available for model training.")
                                if X_train.shape[0] != y_train.shape[0]:
                                    logger.error(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                    raise ValueError(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                if np.any(pd.isnull(X_train)) or np.any(pd.isnull(y_train)):
                                    logger.error("NaN values detected in features or labels.")
                                    raise ValueError("NaN values detected in features or labels.")
                                if len(np.unique(y_train)) < 2:
                                    logger.warning("Skipping model update: only one class present in y_train.")
                                    continue
                                prev_model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                                model = prev_model
                            else:
                                if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                                    logger.error("No data available for model training.")
                                    raise ValueError("No data available for model training.")
                                if X_train.shape[0] != y_train.shape[0]:
                                    logger.error(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                    raise ValueError(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                if np.any(pd.isnull(X_train)) or np.any(pd.isnull(y_train)):
                                    logger.error("NaN values detected in features or labels.")
                                    raise ValueError("NaN values detected in features or labels.")
                                model.fit(X_train, y_train)
                        else:
                            if hasattr(model, "partial_fit"):
                                if len(np.unique(y_train)) < 2:
                                    logger.warning("Skipping model update: only one class present in y_train.")
                                    continue
                                model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                            else:
                                model.fit(X_train, y_train)
                        # Store the fitted model for this ticker
                        if ticker not in fitted_models:
                            fitted_models[ticker] = {}
                        fitted_models[ticker][name] = model
                        save_key = f"models/{self.run_id}/{ticker}_{name}.joblib"
                        save_model_to_s3(model, self.config['s3_bucket'], save_key)
                        preds = model.predict(X_test)

                        acc = accuracy_score(y_test, preds)
                        cm = confusion_matrix(y_test, preds)
                        wins = int((preds == 1).sum())
                        losses = int((preds == 0).sum())
                        total_trades = len(preds)
                        win_rate = (preds == y_test).sum() / total_trades if total_trades > 0 else 0

                        # Simulate PnL: assume +1 for correct, -1 for incorrect (customize for your logic)
                        pnl = ((preds == y_test) * 1 + (preds != y_test) * -1).sum()

                        ticker_results[name] = {
                            "accuracy": acc,
                            "win_rate": win_rate,
                            "trades": total_trades,
                            "wins": wins,
                            "losses": losses,
                            "pnl": pnl,
                            "confusion_matrix": cm.tolist()
                        }

                        # Log results to terminal
                        logging.info(f"[{ticker}][{name}] Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Accuracy: {acc:.2%}, Win Rate: {win_rate:.2%}, PnL: {pnl}")

                    results[ticker] = ticker_results



                summary_rows = []
                for ticker, ticker_results in results.items():
                    for name, metrics in ticker_results.items():
                        summary_rows.append({
                            "Ticker": ticker,
                            "Model": name,
                            "Trades": metrics['trades'],
                            "Wins": metrics['wins'],
                            "Losses": metrics['losses'],
                            "Accuracy": f"{metrics['accuracy']:.2%}",
                            "Win Rate": f"{metrics['win_rate']:.2%}",
                            "PnL": metrics['pnl']
                        })

                summary_df = pd.DataFrame(summary_rows)
                print("\n=== ML Model Performance Summary ===")
                print(summary_df.to_string(index=False))

                # Save the last trained timestamp/index
                if not X_train.empty and "timestamp" in X_train.columns:
                    last_trained_time = X_train["timestamp"].max()
                    with open(LAST_TRAINED_FILE, "w") as f:
                        f.write(str(last_trained_time))

                # --- End ML Model Block ---

                # --- RL PPO Training and Inference Block ---
                logging.warning("RL SYSTEM: Running PPO RL training and inference.")
                # Example: Use the first ticker with enough data for PPO
                for ticker in features:
                    X = feature_store.get(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    if len(X) > 20:  # Ensure enough data
                        split_idx = int(len(X) * 0.8)
                        train_data = X.iloc[:split_idx]
                        test_data = X.iloc[split_idx:]
                        y_train = y.iloc[:split_idx]
                        y_test = y.iloc[split_idx:]
                        regime = detect_regime(train_data['close'])
                        if regime == "high_vol":
                            model_path = f"/home/ubuntu/models/ppo_model_{ticker}_highvol.zip"
                        else:
                            model_path = f"/home/ubuntu/models/ppo_model_{ticker}_normal.zip"
                        self.train_ppo(train_data, ticker, save_path=model_path)
                        self.run_ppo(test_data, ticker)
                        break  # Only run PPO on the first valid ticker for now
                # --- End RL PPO Block ---

                # === Begin Stacking/Meta-Model Block (Fixed) ===
                logging.warning("META-MODEL SYSTEM: Running stacking/meta-model training and prediction.")
                models_from_s3 = load_all_models(self.config['s3_bucket'], self.run_id)

                meta_X = []
                meta_y = []

                meta_signals = None

                for ticker in features:
                    X = feature_store.get(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        y_test = y.iloc[split_idx:]
                    # Collect predictions from all base models
                    preds = []
                    model_dict = models_from_s3.get(ticker, {})
                    # Only proceed if all expected models are present
                    expected_models = ["RandomForest", "PPO"]
                    if not all(m in model_dict for m in expected_models):
                        logging.warning(f"Skipping {ticker} in stacking: missing one or more models ({expected_models})")
                        continue
                    for model_name in expected_models:
                        model = model_dict[model_name]
                        if model_name == "PPO":
                            rl_preds = []
                            env = TradingEnv(X_test)
                            obs, info = env.reset()
                            done = False
                            while not done:
                                action, _ = model.predict(obs, deterministic=True)
                                obs, reward, terminated, truncated, info = env.step(action)
                                rl_preds.append(action)
                                done = terminated or truncated
                            rl_preds = np.array(rl_preds[:len(X_test)]).flatten()
                            preds.append(rl_preds)
                        else:
                            # Drop any datetime or identifier columns from features before prediction
                            cols_to_drop = []
                            for col in X_test.columns:
                                if np.issubdtype(X_test[col].dtype, np.datetime64) or col in ["datetime", "timestamp"]:
                                    cols_to_drop.append(col)
                            if cols_to_drop:
                                X_test_pred = X_test.drop(columns=cols_to_drop)
                            else:
                                X_test_pred = X_test
                                ml_preds = model.predict(X_test_pred)
                                ml_preds = np.array(ml_preds).flatten()
                                preds.append(ml_preds)

                        # Align all predictions to the minimum available length
                        min_len = min(len(p) for p in preds)
                        n_models = len(preds)
                        for i in range(min_len):
                            row = []
                            for p in preds:
                                val = p[i]
                                # Flatten any array or list to scalar
                                if isinstance(val, (np.ndarray, list)):
                                    val = np.asarray(val).flatten()
                                    if val.shape == ():  # scalar
                                        val = val.item()
                                    elif val.shape == (1,):
                                        val = val[0]
                                    else:
                                        logger.error(f"Non-scalar prediction at index {i}: {val} (shape {val.shape})")
                                        logger.error(f"Non-scalar prediction at index {i}: {val} (shape {val.shape})")
                                        raise ValueError(f"Non-scalar prediction at index {i}: {val} (shape {val.shape})")
                                row.append(int(val))
                            # Only append rows with the correct number of model predictions and all scalars
                            if len(row) == n_models and all(isinstance(x, (int, float, np.integer, np.floating)) for x in row):
                                meta_X.append(row)
                                meta_y.append(int(y_test.iloc[i]))
                            else:
                                logging.warning(f"Skipping row {i} for {ticker} due to length/type mismatch: {row}")

                # Final check before fitting meta-model
                if len(meta_X) == 0:
                    logger.error("No valid meta_X rows to fit meta-model. Skipping meta-model training for this run.")
                else:
                    meta_X_arr = np.array(meta_X)
                    logging.info(f"meta_X_arr shape: {meta_X_arr.shape}, dtype: {meta_X_arr.dtype}")
                    logging.info(f"meta_X_arr example row: {meta_X_arr[0]}")
                    meta_model_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/meta_model.joblib")
                    if meta_model_key:
                        prev_meta_model = load_model_from_s3(self.config['s3_bucket'], meta_model_key)
                        if hasattr(prev_meta_model, "partial_fit"):
                            if meta_X_arr.shape[0] == 0 or meta_y.shape[0] == 0:
                                logger.error("No data available for meta-model training.")
                                raise ValueError("No data available for meta-model training.")
                            if meta_X_arr.shape[0] != meta_y.shape[0]:
                                logger.error(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                                raise ValueError(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                            if np.any(pd.isnull(meta_X_arr)) or np.any(pd.isnull(meta_y)):
                                logger.error("NaN values detected in meta-model features or labels.")
                                raise ValueError("NaN values detected in meta-model features or labels.")
                                # Check for at least two unique classes before partial_fit
                                if len(np.unique(meta_y)) < 2:
                                    logger.warning("Skipping model update: only one class present in meta_y.")
                                    continue
                                prev_meta_model.partial_fit(meta_X_arr, meta_y)
                            meta_model = prev_meta_model
                        else:
                            meta_model = LogisticRegression()
                            # Validate meta-model features and labels before training
                            if meta_X_arr.shape[0] == 0 or meta_y.shape[0] == 0:
                                logger.error("No data available for meta-model training.")
                                raise ValueError("No data available for meta-model training.")
                            if meta_X_arr.shape[0] != meta_y.shape[0]:
                                logger.error(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                                raise ValueError(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                            if np.any(pd.isnull(meta_X_arr)) or np.any(pd.isnull(meta_y)):
                                logger.error("NaN values detected in meta-model features or labels.")
                                raise ValueError("NaN values detected in meta-model features or labels.")
                            meta_model.fit(meta_X_arr, meta_y)
                    else:
                        meta_model = LogisticRegression()
                        meta_model.fit(meta_X_arr, meta_y)
                    # Save the updated meta-model
                    save_model_to_s3(meta_model, self.config['s3_bucket'], f"models/{self.run_id}/meta_model.joblib")
                    logging.info("Meta-model (stacking) training complete.")

                    # Use meta-model for final trading signal
                    meta_signals = meta_model.predict(meta_X_arr)
                    logging.info(f"Meta-model signals (first 10): {meta_signals[:10]}")
                # === End Stacking/Meta-Model Block (Fixed) ===

                # === Begin Meta-Model Trading Simulation Block ===
                # For each ticker, simulate trading using meta-model signals

                # We'll assume meta_signals are in the same order as meta_X/meta_y, which are built sequentially for all tickers.
                # We'll split meta_signals back into per-ticker segments for reporting.

                signal_idx = 0
                tickers_with_meta = []
                for ticker in features:
                    X = features[ticker]
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        n = len(X_test)
                        if meta_signals is not None and signal_idx + n <= len(meta_signals):
                            ticker_signals = meta_signals[signal_idx:signal_idx + n]
                            prices = X_test['close'].values
                            position = 0  # if=long, -1=short, 0=flat
                            balance = 100000
                            trade_log = []

                            # Track daily PnL and starting equity
                            current_day = datetime.date.today()
                            starting_equity = balance
                            daily_pnl = 0.0
                            trading_halted_for_day = False

                            for i, signal in enumerate(ticker_signals):
                                # check if a new day has started
                                if datetime.date.today() != current_day:
                                    current_day = datetime.date.today()
                                    feature_store.clear()  # Clear real-time feature store at the start of each new trading day
                                    starting_equity = balance
                                    daily_pnl = 0.0
                                    trading_halted_for_day = False
                                    trades_today = 0

                                prev_position = position

                                # Update daily PnL
                                daily_pnl = balance - starting_equity

                                # Enforce max daily loss
                                if daily_pnl <= -MAX_DAILY_LOSS_PCT * starting_equity:
                                    if not trading_halted_for_day:
                                        print(f"Max daily loss reached ({daily_pnl:.2f}). Halting trading for the day.")
                                        trading_halted_for_day = True

                                # Enforce max trades per day
                                if trades_today >= MAX_TRADES_PER_DAY:
                                    if not trading_halted_for_day:
                                        print(f"Max trades per day reached ({trades_today}). Halting trading for the day.")
                                        trading_halted_for_day = True

                                if trading_halted_for_day:
                                    continue  # Skip trade execution if halted

                                order_size = 1  # Default order size for each trade

                                if signal == 1 and position <= 0:
                                    if abs(position + order_size) <= MAX_POSITION_EXPOSURE:
                                        if order_size <= MAX_ORDER_SIZE:
                                            position += order_size
                                            execute_trade(ib, contract, 'BUY', order_size)
                                            trades_today += 1
                                        else:
                                            print(f"Order size {order_size} exceeds max allowed ({MAX_ORDER_SIZE}). No BUY executed.")
                                    else:
                                        print(f"Position exposure limit reached. Cannot BUY. Current position: {position}")
                                elif signal == -1 and position >= 0:
                                    if abs(position - order_size) <= MAX_POSITION_EXPOSURE:
                                        if order_size <= MAX_ORDER_SIZE:
                                            position -= order_size
                                            execute_trade(ib, contract, 'SELL', order_size)
                                            trades_today += 1
                                        else:
                                            print(f"Order size {order_size} exceeds max allowed ({MAX_ORDER_SIZE}). No SELL executed.")
                                    else:
                                        print(f"Position exposure limit reached. Cannot SELL. Current position: {position}")
                                elif signal == 0:
                                    position = 0
                                # No trade for hold
                            if i > 0:
                                balance += prev_position * (price - prices[i-1])
                            print(f"\n=== Meta-Model Trading Results for {ticker} ===")
                            print(f"Final balance: {balance:.2f}")
                            print(f"Total trades: {len([t for t in trade_log if t[0] != 'hold'])}")
                            print(f"Trade log (first 10): {trade_log[:10]}")
                            signal_idx += n
                            tickers_with_meta.append(ticker)
                        else:
                            print(f"\n=== Meta-Model Trading Results for {ticker} ===")
                            print("No meta-model predictions available for this ticker (skipped in stacking).")
                            if meta_signals is not None:
                                print(f"meta_signals length: {len(meta_signals)}, tickers with meta: {tickers_with_meta}")
                            else:
                                print("meta_signals is None (meta-model was not trained this run).")
                # === End Meta-Model Trading Simulation Block ===

                # === Begin Ensemble Voting Block ===
                logging.warning("ENSEMBLE SYSTEM: Running ensemble voting for all tickers.")
                # For each ticker, collect predictions from all ML models and RL model, then vote

                ensemble_results = {}

                for ticker in features:
                    X = feature_store.get(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        y_test = y.iloc[split_idx:]

                        required_models = ["SGDClassifier", "Perceptron", "PassiveAggressive", "RandomForest"]
                        if not all(name in fitted_models.get(ticker, {}) for name in required_models):
                            logging.warning(f"Not all models fitted for {ticker}, skipping ensemble voting for this ticker.")
                            continue

                        # Drop any datetime or identifier columns from features before prediction
                        cols_to_drop = []
                        for col in X_test.columns:
                            if np.issubdtype(X_test[col].dtype, np.datetime64) or col in ["datetime", "timestamp"]:
                                cols_to_drop.append(col)
                        if cols_to_drop:
                            X_test_pred = X_test.drop(columns=cols_to_drop)
                        else:
                            X_test_pred = X_test

                        # Get ML model predictions
                        ml_preds = {}
                        for name, model in fitted_models.get(ticker, {}).items():
                            ml_preds[name] = model.predict(X_test_pred)

                        # Get RL model predictions (PPO)
                        rl_preds = []
                        env = TradingEnv(X_test)
                        obs, info = env.reset()
                        done = False
                        while not done:
                            action, _ = self.ppo_model.predict(obs, deterministic=True)
                            rl_preds.append(action)
                            obs, reward, terminated, truncated, info = env.step(action)
                            done = terminated or truncated
                        rl_preds = np.array(rl_preds[:len(X_test)]).flatten()
                        preds.append(rl_preds)

                        # Align all predictions to the minimum available length
                        min_len = min(
                            len(ml_preds["RandomForest"]),
                            len(rl_preds)
                        )

                        final_signals = []
                        for i in range(min_len):
                            votes = [
                                ml_preds["RandomForest"][i],
                                rl_preds[i]
                            ]
                            # Majority vote (mode)
                            votes = [int(v) if isinstance(v, np.ndarray) else v for v in votes]
                            final_signal = max(set(votes), key=votes.count)
                            final_signals.append(final_signal)

                        # Store and log ensemble results
                        ensemble_results[ticker] = final_signals
                        logging.info(f"[{ticker}] Ensemble Voting Signals: {final_signals[:10]}... (showing first 10)")

                # === End Ensemble Voting Block ===

                import os

                # Save and upload each ticker's features
                for ticker, df in features.items():
                    # Create a unique filename per ticker and run
                    local_csv_path = f"{ticker}_features_{self.run_id}.csv"
                    df.to_csv(local_csv_path, index=False)
                    # S3 key: organize by run and ticker for auditability
                    s3_key = f"features/{self.run_id}/{ticker}_features.csv"
                    upload_file_to_s3(local_csv_path, self.config['s3_bucket'], s3_key)
                    # Optional: remove local file after upload to save space
                    os.remove(local_csv_path)
                logging.info("Features engineered successfully.")

                # RL model inference and trading logic
                # (Your original RL model code remains here, unchanged)
                # Example:
                # actions = self.rl_model.predict(features)
                # self.execute_trades(actions)

                log_trade_results(f"pipeline_success: run_id={self.run_id}")
                self.notify("Pipeline run completed successfully.")
                break  # Success, exit retry loop

            except Exception as e:
                retries += 1
                logging.warning(f"Error in pipeline loop (attempt {retries}): {e}")
                log_trade_results(f"pipeline_retry: run_id={self.run_id}, attempt={retries}, error={str(e)}, traceback={traceback.format_exc()}")
                self.notify(f"Pipeline error (attempt {retries}): {e}")
                prev_key = get_latest_model_key(self.config['s3_bucket'], "models/", exclude_run_id=self.run_id)
                if prev_key:
                    model = load_model_from_s3(self.config['s3_bucket'], prev_key)
                    logging.info(f"Rolled back to previous model: {prev_key}")
                    # RL model rollback
                    if prev_key.endswith("_ppo.zip"):
                        from stable_baselines3 import PPO
                        local_ppo_path = "/tmp/" + os.path.basename(prev_key)
                        s3 = boto3.client("s3")
                        s3.download_file(self.config['s3_bucket'], prev_key, local_ppo_path)
                        self.ppo_model = PPO.load(local_ppo_path)
                        logging.info("self.ppo_model has been set to the rolled-back PPO model.")
                        os.remove(local_ppo_path)
                    # ML model rollback
                    elif prev_key.endswith(".joblib"):
                        import re
                        match = re.search(r"models/[^/]+/([^_]+)_([^/]+)\.joblib", prev_key)
                        if match:
                            ticker, model_name = match.groups()
                            if not hasattr(self, "ml_models"):
                                self.ml_models = {}
                            if ticker not in self.ml_models:
                                self.ml_models[ticker] = {}
                            self.ml_models[ticker][model_name] = model
                            logging.info(f"Rolled back ML model for {ticker} - {model_name}.")
                else:
                    logger.error("No previous model found for rollback.")
                if retries >= self.max_retries:
                    logger.error("Max retries reached. Failing pipeline.")
                    raise
                time.sleep(self.retry_delay)

    # Extensibility: add hooks for notifications, multi-asset, etc.
    def add_notification_hook(self, hook_fn):
        self.notification_hook = hook_fn

    def notify(self, message):
        if self.notification_hook:
            self.notification_hook(message)
        else:
            logging.info(f"Notification: {message}")

    def train_ppo(self, train_data, ticker, save_path=None):
        env = TradingEnv(train_data)
        check_env(env)
        ppo_logger = PPOTrainingLogger()
        ppo_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/{ticker}_ppo.zip")
        local_ppo_path = "/tmp/ppo_model.zip"
        if ppo_key:
            # Download the PPO model from S3 to local path
            s3 = boto3.client("s3")
            s3.download_file(self.config['s3_bucket'], ppo_key, local_ppo_path)
            model = PPO.load(local_ppo_path)
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=10000, callback=ppo_logger)
            os.remove(local_ppo_path)
        else:
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=10000, callback=ppo_logger)
        self.ppo_model = model
        ppo_logger.print_summary()
        ppo_key = f"models/{self.run_id}/{ticker}_ppo.zip"
        if save_path is None:
            save_path = "/tmp/ppo_model.zip"
        model.save(save_path)
        s3 = boto3.client("s3")
        s3.upload_file("/tmp/ppo_model.zip", self.config['s3_bucket'], ppo_key)
        os.remove("/tmp/ppo_model.zip")
        logging.info(f"Retraining event: PPO model for {ticker} retrained and saved as {ppo_key}")
        print("PPO training complete.")

    def run_ppo(self, test_data, ticker):
        from stable_baselines3 import PPO
        env = TradingEnv(test_data)
        regime = detect_regime(test_data['close'])
        if regime == "high_vol":
            model_path = f"/home/ubuntu/models/ppo_model_{ticker}_highvol.zip"
        else:
            model_path = f"/home/ubuntu/models/ppo_model_{ticker}_normal.zip"
        if not os.path.exists(model_path):
            model_path = f"/home/ubuntu/models/ppo_model_{ticker}.zip"
        if not os.path.exists(model_path):
            logging.warning(f"No PPO model found for {ticker} at {model_path}; skipping PPO run.")
            return 0.0
        self.ppo_model = PPO.load(model_path)
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"PPO test run total reward: {total_reward}")
        return total_reward

    def train_ensemble_ppo(self, train_data, n_models=3):
        self.ensemble_models = []
        for i in range(n_models):
            env = TradingEnv(train_data)
            model = PPO("MlpPolicy", env, verbose=0, seed=i)
            model.learn(total_timesteps=10000)
            self.ensemble_models.append(model)
        print(f"Trained ensemble of {n_models} PPO models.")

    def run_ensemble_ppo(self, test_data):
        env = TradingEnv(test_data)
        obs, info = env.reset()
        done = False
        total_reward = 0
        actions = []
        while not done:
            action_votes = [model.predict(obs)[0] for model in self.ensemble_models]
            majority_action = max(set(action_votes), key=action_votes.count)
            confidence = action_votes.count(majority_action) / len(action_votes)
            if confidence < 0.6:
                majority_action = 0  # Hold if not confident
            obs, reward, terminated, truncated, info = env.step(majority_action)
            done = terminated or truncated
            total_reward += reward

# Example config for testing
default_config = {
    "tickers": ["ES=F", "NQ=F", "GC=F", "6E=F"],
    "use_mock_data": false,
    "feature_params": {"window": 20, "indicators": ["sma", "ema", "rsi"]},
    "log_path": "logs/pipeline.log",
    "s3_bucket": "omega-singularity-ml",
    "max_retries": 3,
    "retry_delay": 10,
    # AI_ASSISTANT_TODO: Set to False before enabling live IBKR data!
    "disable_drift_detection": True,  # TEMP: Disable drift detection for mock data testing
    "drift_detection_threshold": 10.0  # Relaxed for mock data; set to 3.0 for production
}

import time

if __name__ == "__main__":
    while True:
        pipeline = RLTradingPipeline(default_config)
        pipeline.run()
        # Sleep for 2 minutes (adjust as needed)
        time.sleep(120)

market_data_adapter.disconnect()
