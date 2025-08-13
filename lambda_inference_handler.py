import boto3
import pandas as pd
import joblib
import json
import os
import logging
import numpy as np
from datetime import datetime
from audit_logging_utils import audit_event  # Added for audit logging

# --- CONFIG ---
S3_BUCKET = os.environ.get("S3_BUCKET", "omega-singularity-ml")
MODEL_PREFIX = "models/"
DATA_PREFIX = os.environ.get("S3_DATA_PREFIX", "market_data/")
OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "predictions/")
TMP_MODEL_PATH = "/tmp/rf_model.joblib"
TMP_FEATURES_PATH = "/tmp/features.json"

# --- LOGGING ---
logger = logging.getLogger("OmegaSingularityInference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)

# --- Feature Mapping (as before) ---
def ohlcv_to_required_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['close', 'high', 'low']:
        if col not in df.columns:
            raise ValueError("OHLCV columns (close, high, low) missing from input DataFrame.")
    df['feature1'] = np.log(df['close'] / df['close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
    df['feature2'] = ((df['high'] - df['low']) / df['close']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    required_base = ['feature1', 'feature2']
    for col in required_base:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from input data.")
    df['feature_sum'] = df['feature1'] + df['feature2']

        return df

def lambda_handler(event, context):
    s3 = boto3.client("s3")
    # Parse event for S3 data key
    data_key = event.get("data_key")
    if not data_key:
        logger.error("No data_key provided in event.")
        return {"status": "error", "message": "No data_key provided."}

    # Download data from S3
    local_data_path = "/tmp/input_data.csv"
    s3.download_file(S3_BUCKET, data_key, local_data_path)
    df = pd.read_csv(local_data_path)
    df = ohlcv_to_required_features(df)
    df = feature_engineering(df)

    # Load model
    model_key = event.get("model_key", f"{MODEL_PREFIX}rf_model.joblib")
    s3.download_file(S3_BUCKET, model_key, TMP_MODEL_PATH)
    model = joblib.load(TMP_MODEL_PATH)

    # Predict
    X = df[['feature1', 'feature2', 'feature_sum']]
    predictions = model.predict(X)
    df['prediction'] = predictions

    # Save predictions to S3
    now_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_key = f"{OUTPUT_PREFIX}pred_{os.path.basename(data_key).replace('.csv', '')}_{now_str}.csv"
    local_output_path = "/tmp/predictions.csv"
    df.to_csv(local_output_path, index=False)
    s3.upload_file(local_output_path, S3_BUCKET, output_key)
    logger.info(f"Uploaded predictions to s3://{S3_BUCKET}/{output_key}")
    # Audit event after successful prediction upload
    audit_event("prediction_finished", {"input_data_key": data_key, "output_data_key": output_key, "records": len(df)}, S3_BUCKET, "audit_logs/")

    return {
        "status": "success",
        "output_key": output_key,
        "records": len(df)
    }
