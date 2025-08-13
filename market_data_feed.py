import time
import logging
import boto3
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

# Configuration
TICKERS = ["ES=F", "NQ=F", "AUDUSD=X", "GBPUSD=X", "EURUSD=X", "XAUUSD=X", "GC=F"]  # Yahoo Finance symbols
FETCH_INTERVAL_MINUTES = 15
S3_BUCKET = "omega-singularity-ml"
S3_BASE_PATH = "market_data"
LOG_FILE = "/var/log/market_data_feed.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def fetch_and_store_ticker(ticker, s3_client):
    try:
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        timestamp = now.strftime("%Y%m%dT%H%M%SZ")
        # Fetch last 16 minutes to ensure overlap
        df = yf.download(ticker, period="16m", interval="1m")
        if df.empty:
            logging.warning(f"No data returned for {ticker} at {timestamp}")
            return
        df.reset_index(inplace=True)
        df["ticker"] = ticker
        # Save to CSV
        filename = f"market_data_{ticker.replace('=','')}_{timestamp}.csv"
        local_path = f"/tmp/{filename}"
        df.to_csv(local_path, index=False)
        # Upload to S3
        s3_key = f"{S3_BASE_PATH}/{ticker.replace('=','')}/{date_path}/{filename}"
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        logging.info(f"Uploaded {ticker} data to s3://{S3_BUCKET}/{s3_key}")
        os.remove(local_path)
    except Exception as e:
        logging.error(f"Error fetching/storing data for {ticker}: {e}")

def main():
    s3_client = boto3.client("s3")
    logging.info("Market data feed started.")
    while True:
        for ticker in TICKERS:
            fetch_and_store_ticker(ticker, s3_client)
        logging.info(f"Sleeping for {FETCH_INTERVAL_MINUTES} minutes.")
        time.sleep(FETCH_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main()
