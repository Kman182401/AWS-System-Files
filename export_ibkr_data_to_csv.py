import os
import time
import pandas as pd
import logging
from datetime import datetime, timedelta

from market_data_ibkr_adapter import IBKRIngestor
from market_data_config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_SYMBOLS
)

DATA_DIR = "/home/ubuntu/data"
CHUNK_DAYS = 30
SLEEP_SECS = 1.0
WHAT_TO_SHOW = ['TRADES', 'MIDPOINT']
USE_RTH = False  # all-hours

# Logging setup
logger = logging.getLogger("ibkr_export")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(h)

def get_last_timestamp(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        ts_col = 'timestamp' if 'timestamp' in df.columns else ('date' if 'date' in df.columns else None)
        if ts_col is None:
            return None
        return pd.to_datetime(df[ts_col]).max()
    except Exception:
        return None

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def export_ticker(ticker: str, ibkr: IBKRIngestor):
    summary = []
    logger.info(f"=== Exporting {ticker} ===")
    for wts in WHAT_TO_SHOW:
        csv_path = os.path.join(DATA_DIR, f"{ticker}_{wts}.csv")
        ensure_dir(os.path.dirname(csv_path))
        last_ts = get_last_timestamp(csv_path)
        end_dt = pd.Timestamp.now()
        start_dt = (pd.to_datetime(last_ts) + pd.Timedelta(minutes=1)) if last_ts is not None else (end_dt - pd.Timedelta(days=730))
        current_start = start_dt
        all_chunks = []

        while current_start < end_dt:
            current_end = min(current_start + pd.Timedelta(days=CHUNK_DAYS), end_dt)
            duration_str = f"{(current_end - current_start).days} D"
            try:
                df = ibkr.fetch_data(
                    ticker,
                    duration=duration_str,
                    barSize='1 min',
                    whatToShow=wts,
                    useRTH=USE_RTH,
                    formatDate=1,
                    endDateTime=current_end.strftime('%Y%m%d %H:%M:%S'),
                    asof=current_end.to_pydatetime()
                )
                if df is not None and not df.empty:
                    # Trim to the requested window, just in case
                    df = df[(df['timestamp'] >= current_start) & (df['timestamp'] <= current_end)]
                    all_chunks.append(df)
                    logger.info(f"[OK] {ticker} {wts} {current_start} -> {current_end} ({len(df)} rows)")
                else:
                    logger.info(f"[NO DATA] {ticker} {wts} {current_start} -> {current_end}")
            except Exception as e:
                logger.warning(f"[ERROR] {ticker} {wts} {current_start} -> {current_end}: {e}")
            current_start = current_end
            time.sleep(SLEEP_SECS)

        rows_written = 0
        if all_chunks:
            df_all = pd.concat(all_chunks, ignore_index=True)
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                if 'timestamp' in df_old.columns:
                    df_old['timestamp'] = pd.to_datetime(df_old['timestamp'])
                elif 'date' in df_old.columns:
                    df_old = df_old.rename(columns={'date': 'timestamp'})
                    df_old['timestamp'] = pd.to_datetime(df_old['timestamp'])
                df_all = pd.concat([df_old, df_all], ignore_index=True)
            df_all = df_all.drop_duplicates(subset='timestamp').sort_values('timestamp')
            df_all.to_csv(csv_path, index=False)
            rows_written = len(df_all)
            logger.info(f"Saved {ticker} {wts}: {rows_written} total rows -> {csv_path}")
            first_ts = df_all['timestamp'].min()
            last_ts = df_all['timestamp'].max()
            summary.append((wts, rows_written, first_ts, last_ts))
        else:
            logger.info(f"No new data for {ticker} {wts}")
            summary.append((wts, 0, None, None))

    # Pretty summary
    logger.info(f"=== Summary for {ticker} ===")
    for wts, rows, fts, lts in summary:
        if rows > 0:
            logger.info(f"{wts:9s} | rows: {rows:8d} | {fts} -> {lts}")
        else:
            logger.info(f"{wts:9s} | rows: {rows:8d} | (no data)")
    logger.info("")

def main():
    ibkr = IBKRIngestor(host=IBKR_HOST, port=IBKR_PORT, clientId=IBKR_CLIENT_ID)
    for ticker in IBKR_SYMBOLS:
        export_ticker(ticker, ibkr)

if __name__ == "__main__":
    main()
