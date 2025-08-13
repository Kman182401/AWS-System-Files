import requests
import pandas as pd

API_KEY = "R6gHRWFkPCg34CavO7S9gp1Jjdw5O0lk78XBxN7D"
BASE_URL = "https://api.marketaux.com/v1/news/all"

def fetch_marketaux_news(
    symbols=None,
    entity_types=None,
    countries=None,
    query=None,
    published_after=None,
    published_before=None,
    limit=20,
    language="en",
    page=1,
    sentiment_gte=None,
    sentiment_lte=None,
    min_match_score=None,
    filter_entities=None,
    must_have_entities=None
):
    params = {
        "api_token": API_KEY,
        "limit": limit,
        "language": language,
        "page": page,
    }
    if symbols is not None:
        params["symbols"] = ",".join(symbols)
    if entity_types is not None:
        params["entity_types"] = entity_types
    if countries is not None:
        params["countries"] = countries
    if query is not None:
        params["search"] = query
    if published_after is not None:
        params["published_after"] = published_after
    if published_before is not None:
        params["published_before"] = published_before
    if sentiment_gte is not None:
        params["sentiment_gte"] = sentiment_gte
    if sentiment_lte is not None:
        params["sentiment_lte"] = sentiment_lte
    if min_match_score is not None:
        params["min_match_score"] = min_match_score
    if filter_entities is not None:
        params["filter_entities"] = filter_entities
    if must_have_entities is not None:
        params["must_have_entities"] = must_have_entities

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])

def normalize_marketaux_news(news_list):
    records = []
    for item in news_list:
        tickers = []
        if "entities" in item and item["entities"]:
            for ent in item["entities"]:
                if "symbol" in ent and ent["symbol"]:
                    tickers.append(ent["symbol"])
        records.append({
            "title": item.get("title"),
            "summary": item.get("description") or item.get("snippet"),
            "published_at": item.get("published_at"),
            "tickers": tickers,
            "entities": item.get("entities"),
            "sentiment_score": item.get("sentiment_score"),
            "source": item.get("source"),
            "url": item.get("url"),
            "keywords": item.get("keywords"),
        })
    df = pd.DataFrame(records)
    if not df.empty and "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    return df
