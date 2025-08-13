import boto3
import json
import pandas as pd
from io import BytesIO
import re
from datetime import datetime
from dateutil import parser as date_parser
import spacy

# Load spaCy's large English model (run 'python -m spacy download en_core_web_lg' if not already installed)
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")

# Example mapping from entity/sector to tickers (customize as needed)
ENTITY_TICKER_MAP = {
    "Federal Reserve": ["ES1!", "NQ1!", "GBPUSD", "EURUSD", "AUDUSD"],
    "interest rates": ["ES1!", "NQ1!", "GBPUSD", "EURUSD", "AUDUSD"],
    "inflation": ["ES1!", "NQ1!", "GC", "GBPUSD", "EURUSD", "AUDUSD"],
    "gold": ["GC"],
    "technology": ["NQ1!"],
    "bank": ["ES1!", "NQ1!"],
    # Add more mappings as needed
}

def load_all_news_from_s3(bucket: str, prefix: str = "news/") -> pd.DataFrame:
    """
    Loads all news JSON files from S3 under the given prefix into a pandas DataFrame.
    Each row is a news article.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    news_records = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                file_obj = s3.get_object(Bucket=bucket, Key=key)
                news_json = json.loads(file_obj["Body"].read().decode("utf-8"))
                news_records.append(news_json)
    if not news_records:
        return pd.DataFrame()  # Return empty DataFrame if no news found
    return pd.DataFrame(news_records)

def advanced_entity_and_impact_extraction(text):
    """
    Extracts entities and maps them to tickers based on context and learned mapping.
    Returns a dict: { "entities": [...], "relevant_tickers": [...] }
    """
    if not isinstance(text, str):
        return {"entities": [], "relevant_tickers": []}
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    relevant_tickers = set()
    for ent in entities:
        for key, tickers in ENTITY_TICKER_MAP.items():
            if key.lower() in ent.lower():
                relevant_tickers.update(tickers)
    # Also check for direct mentions of tickers
    for ticker in ["ES1!", "NQ1!", "GBPUSD", "EURUSD", "AUDUSD", "GC"]:
        if ticker in text:
            relevant_tickers.add(ticker)
    return {"entities": entities, "relevant_tickers": list(relevant_tickers)}

def engineer_news_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds feature columns to the news DataFrame:
    - main_topic: str (first keyword or topic)
    - entities: list of str (NER extraction)
    - relevant_tickers: list of str (mapped from entities and context)
    - time_published_dt: datetime (parsed and timezone-aware)
    """
    # Sentiment score
    if "overall_sentiment_score" in news_df.columns:
        news_df["sentiment_score"] = pd.to_numeric(news_df["overall_sentiment_score"], errors="coerce")
    else:
        news_df["sentiment_score"] = None  # Placeholder for custom sentiment

    # Main topic/keyword
    if "topics" in news_df.columns:
        news_df["main_topic"] = news_df["topics"].apply(lambda x: x[0]["topic"] if isinstance(x, list) and x else None)
    elif "keywords" in news_df.columns:
        news_df["main_topic"] = news_df["keywords"].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    else:
        news_df["main_topic"] = None

    # Advanced entity and impact extraction
    if "summary" in news_df.columns:
        impact_results = news_df["summary"].apply(advanced_entity_and_impact_extraction)
        news_df["entities"] = impact_results.apply(lambda x: x["entities"])
        news_df["relevant_tickers"] = impact_results.apply(lambda x: x["relevant_tickers"])
    else:
        news_df["entities"] = None
        news_df["relevant_tickers"] = None

    # Timestamp normalization
    def parse_time(ts):
        try:
            return date_parser.parse(ts)
        except Exception:
            return pd.NaT

    if "time_published" in news_df.columns:
        news_df["time_published_dt"] = news_df["time_published"].apply(parse_time)
    else:
        news_df["time_published_dt"] = pd.NaT

    return news_df

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a transformer model for embeddings (first time will download)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_news_embeddings(news_df: pd.DataFrame, text_col: str = None) -> np.ndarray:
    """
    Computes embeddings for all news articles in the DataFrame.
    Returns a numpy array of shape (n_articles, embedding_dim).
    Automatically selects the first available text column if not specified.
    """
    # Try to auto-detect the main text column if not specified
    if text_col is None:
        for candidate in ["summary", "content", "body", "title", "text"]:
            if candidate in news_df.columns:
                text_col = candidate
                break
        else:
            raise KeyError("No suitable text column found in news_df. Checked: summary, content, body, title, text.")
    texts = news_df[text_col].fillna("").tolist()
    embeddings = sentence_model.encode(texts, show_progress_bar=True)
    return embeddings

def find_similar_news(
    news_df: pd.DataFrame,
    historical_impactful_news_df: pd.DataFrame,
    embeddings: np.ndarray,
    historical_embeddings: np.ndarray,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Finds the top_k most similar news articles to new_text in news_df.
    Returns a DataFrame of the most similar articles.
    """
    new_embedding = sentence_model.encode([new_text])
    similarities = cosine_similarity(new_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    similar_news = news_df.iloc[top_indices].copy()
    similar_news["similarity"] = similarities[top_indices]
    return similar_news
