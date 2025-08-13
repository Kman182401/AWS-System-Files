import pandas as pd
import numpy as np
import pandas as pd

def analyze_price_impact(
    similar_news_df: pd.DataFrame,
    ticker: str,
    price_df: pd.DataFrame,
    time_col: str = "time_published_dt",
    price_col: str = "close",
    window: str = "1D"
) -> pd.DataFrame:
    """
    For each news event in similar_news_df, calculate the price change in the specified window after the event.
    Returns a DataFrame with news, event time, and price impact.
    """
    impacts = []
    for _, row in similar_news_df.iterrows():
        event_time = row[time_col]
        if pd.isnull(event_time):
            continue
        # Find the price at the event time (or next available)
        price_at_event = price_df[price_df.index >= event_time][price_col].iloc[0] if not price_df[price_df.index >= event_time].empty else None
        # Find the price at the end of the window
        end_time = event_time + pd.Timedelta(window)
        price_at_end = price_df[price_df.index >= end_time][price_col].iloc[0] if not price_df[price_df.index >= end_time].empty else None
        if price_at_event is not None and price_at_end is not None:
            price_change = (price_at_end - price_at_event) / price_at_event
            impacts.append({
                "summary": row.get("summary", ""),
                "event_time": event_time,
                "price_at_event": price_at_event,
                "price_at_end": price_at_end,
                "price_change": price_change,
                "similarity": row.get("similarity", None)
            })
    return pd.DataFrame(impacts)

def is_price_move_significant(price_series, event_time, lookback=60, post_event=5, z_thresh=2.0):
    """
    Determines if the price move after a news event is statistically significant.
    Args:
        price_series (pd.Series): Price series indexed by datetime.
        event_time (pd.Timestamp): Timestamp of the news event.
        lookback (int): Number of periods before event to use for mean/std calculation.
        post_event (int): Number of periods after event to measure move.
        z_thresh (float): Z-score threshold for significance.
    Returns:
        dict: {
            'is_significant': bool,
            'z_score': float,
            'price_move': float,
            'pre_event_mean': float,
            'pre_event_std': float
        }
    """
    # Ensure event_time is in the index
    if event_time not in price_series.index:
        return {'is_significant': False, 'z_score': np.nan, 'price_move': np.nan, 'pre_event_mean': np.nan, 'pre_event_std': np.nan}

    # Get pre-event window
    pre_event_idx = price_series.index.get_loc(event_time)
    if pre_event_idx < lookback:
        return {'is_significant': False, 'z_score': np.nan, 'price_move': np.nan, 'pre_event_mean': np.nan, 'pre_event_std': np.nan}

    pre_event_prices = price_series.iloc[pre_event_idx - lookback:pre_event_idx]
    pre_event_returns = pre_event_prices.pct_change().dropna()
    pre_event_mean = pre_event_returns.mean()
    pre_event_std = pre_event_returns.std()

    # Get post-event window
    post_event_end_idx = min(pre_event_idx + post_event, len(price_series) - 1)
    post_event_price = price_series.iloc[post_event_end_idx]
    event_price = price_series.iloc[pre_event_idx]
    price_move = (post_event_price - event_price) / event_price

    # Calculate z-score
    if pre_event_std == 0 or np.isnan(pre_event_std):
        z_score = np.nan
        is_significant = False
    else:
        z_score = (price_move - pre_event_mean) / pre_event_std
        is_significant = abs(z_score) >= z_thresh

    return {
        'is_significant': is_significant,
        'z_score': z_score,
        'price_move': price_move,
        'pre_event_mean': pre_event_mean,
        'pre_event_std': pre_event_std
    }
