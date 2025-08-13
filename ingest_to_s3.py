import json
from io import BytesIO
import argparse
import pandas as pd
import boto3
import os
from datetime import datetime
from audit_logging_utils import log_trade_results
from market_data_config import S3_BUCKET_NAME

def upload_file_to_s3(file_path, bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, bucket, key)
        log_trade_results(f"S3_UPLOAD_SUCCESS: bucket={bucket}, key={key}")
        print(f"Upload successful: s3://{bucket}/{key}")
    except Exception as e:
        log_trade_results(f"S3_UPLOAD_FAILURE: bucket={bucket}, key={key}, error={str(e)}")
        print(f"Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ingest a CSV file to S3.")
    parser.add_argument('--file', required=True, help='Path to the input CSV file')
    parser.add_argument('--bucket', required=False, default=S3_BUCKET_NAME, help='S3 bucket name')
    parser.add_argument('--prefix', required=False, default='', help='S3 key prefix (optional)')
    args = parser.parse_args()

    file_path = args.file
    bucket = args.bucket
    prefix = args.prefix

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        log_trade_results(f"FILE_NOT_FOUND: file={file_path}")
        return

    filename = os.path.basename(file_path)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    s3_key = f"{prefix}{filename.rsplit('.', 1)[0]}_{timestamp}.csv"

    upload_file_to_s3(file_path, bucket, s3_key)

def ingest_to_s3(data, s3_path, s3_client=None, bucket=None):
    """
    Ingests a Python dict or list as a JSON file directly to S3.
    Args:
        data: The data to upload (dict or list).
        s3_path: The S3 key (path) to upload to.
        s3_client: Optional boto3 S3 client. If None, a new client is created.
        bucket: S3 bucket name. If None, uses S3_BUCKET_NAME from config.
    """
    if s3_client is None:
        s3_client = boto3.client('s3')
    if bucket is None:
        bucket = S3_BUCKET_NAME

    try:
        json_bytes = json.dumps(data).encode('utf-8')
        s3_client.upload_fileobj(BytesIO(json_bytes), bucket, s3_path)
        log_trade_results(f"S3_UPLOAD_SUCCESS: bucket={bucket}, key={s3_path}")
        print(f"Upload successful: s3://{bucket}/{s3_path}")
    except Exception as e:
        log_trade_results(f"S3_UPLOAD_FAILURE: bucket={bucket}, key={s3_path}, error={str(e)}")
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
