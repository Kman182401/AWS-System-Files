import boto3
import pandas as pd
from datetime import datetime
import json

from audit_logging_utils import audit_event
from market_data_config import S3_BUCKET_NAME

def get_sample_data():
    """
    Replace this with real data source as needed.
    """
    now = datetime.utcnow()
    data = {
        "timestamp": [now],
        "price": [100.0],
        "volume": [10]
    }
    return pd.DataFrame(data)

def upload_to_s3(df, bucket, key):
    s3 = boto3.client('s3')
    csv_buffer = df.to_csv(index=False)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer)
        audit_event("S3_UPLOAD_SUCCESS", {"bucket": bucket, "key": key})
        print(f"Upload successful: s3://{bucket}/{key}")
    except Exception as e:
        audit_event("S3_UPLOAD_FAILURE", {"bucket": bucket, "key": key, "error": str(e)})
        print(f"Upload failed: {e}")

def main():
    bucket = S3_BUCKET_NAME
    key = f"test_data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df = get_sample_data()
    upload_to_s3(df, bucket, key)

if __name__ == "__main__":
    main()
