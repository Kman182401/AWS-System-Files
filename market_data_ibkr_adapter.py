from ib_insync import IB, util, Contract, Future
import pandas as pd
from datetime import datetime
from typing import Optional

def get_ibkr_front_month_contract(
    ib: IB,
    symbol: str,
    exchange: str,
    currency: str,
    asof: Optional[datetime] = None,
    tradingClass: Optional[str] = None
):
    """
    Return the nearest non-expired futures contract for the given symbol as of a reference date.
    Supports specifying tradingClass (e.g., '6E','6B','6A') for FX futures that require underlying
    symbol + tradingClass. Tries CME/GLOBEX synonyms automatically and qualifies the contract.
    """
    ex = exchange.upper()
    if ex in ('GLOBEX', 'CME'):
        exchange_candidates = ['CME', 'GLOBEX']  # prefer CME but try both
    else:
        exchange_candidates = [exchange]

    ref_ymd = int((asof or datetime.utcnow()).strftime('%Y%m%d'))

    for ex_name in exchange_candidates:
        # Prefer querying with tradingClass when provided (FX futures case)
        if tradingClass:
            template = Future(symbol=symbol, exchange=ex_name, currency=currency, tradingClass=tradingClass)
        else:
            template = Future(symbol=symbol, exchange=ex_name, currency=currency)

        details = ib.reqContractDetails(template)
        if not details:
            continue

        def expiry_int(cd):
            exp = cd.contract.lastTradeDateOrContractMonth or ''
            exp = (exp + '01') if len(exp) == 6 else exp
            try:
                return int(exp)
            except ValueError:
                return 99999999

        # Filter to the right tradingClass when requested
        if tradingClass:
            details = [d for d in details if getattr(d.contract, 'tradingClass', None) == tradingClass]

        if not details:
            continue

        details.sort(key=expiry_int)

        for cd in details:
            exp_i = expiry_int(cd)
            if exp_i >= ref_ymd and getattr(cd.contract, 'secType', 'FUT') == 'FUT':
                ib.qualifyContracts(cd.contract)
                return cd.contract

    raise RuntimeError(
        f"No valid (non-expired) futures contract found for {symbol} "
        f"on {exchange_candidates} as of {ref_ymd} (tradingClass={tradingClass})."
    )

class IBKRIngestor:
    def __init__(self, host='127.0.0.1', port=4002, clientId=101):
        self.ib = IB()
        self.connected = False
        self.ib.connect(host, port, clientId=clientId)
        self.connected = True

    def fetch_data(self, ticker, duration='1 D', barSize='1 min', whatToShow='TRADES',
                   useRTH=False, formatDate=1, endDateTime='', asof: Optional[datetime] = None):
        if not self.connected:
            raise RuntimeError(f"IBKR not connected. Cannot fetch data for {ticker}.")
        contract = self._get_contract(ticker, asof=asof)
        if contract is None:
            raise RuntimeError(f"No contract available for {ticker}.")
        print(f"Requesting contract: {contract}")  # Debug log
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=endDateTime,
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=formatDate
            )
            df = util.df(bars)
            if df is not None and not df.empty:
                # Ensure a 'timestamp' column exists and is datetime
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'timestamp'})
                elif df.index.name == 'date':
                    df = df.reset_index().rename(columns={'date': 'timestamp'})
                else:
                    df['timestamp'] = df.index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Keep only relevant columns
                keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = df[[c for c in keep_cols if c in df.columns]]
                return df
            else:
                raise RuntimeError(f"No data returned for {ticker} (contract: {contract}).")
        except Exception as e:
            raise RuntimeError(f"Exception during IBKR data fetch for {ticker}: {e}")

    def _get_contract(self, ticker, asof: Optional[datetime] = None):
        # Map TradingView-style symbols to IB futures templates; use as-of date for backfills
        if ticker == 'ES1!':
            return get_ibkr_front_month_contract(self.ib, 'ES', 'CME', 'USD', asof=asof)
        elif ticker == 'NQ1!':
            return get_ibkr_front_month_contract(self.ib, 'NQ', 'CME', 'USD', asof=asof)
        elif ticker == 'XAUUSD':
            # Use COMEX Gold (GC) as the futures proxy for XAUUSD
            return get_ibkr_front_month_contract(self.ib, 'GC', 'COMEX', 'USD', asof=asof)
        elif ticker == 'EURUSD':
            # FX futures via CME with explicit tradingClass for currency futures
            return get_ibkr_front_month_contract(self.ib, 'EUR', 'CME', 'USD', asof=asof, tradingClass='6E')
        elif ticker == 'AUDUSD':
            return get_ibkr_front_month_contract(self.ib, 'AUD', 'CME', 'USD', asof=asof, tradingClass='6A')
        elif ticker == 'GBPUSD':
            return get_ibkr_front_month_contract(self.ib, 'GBP', 'CME', 'USD', asof=asof, tradingClass='6B')
        else:
            raise ValueError(f"Unknown ticker: {ticker}.")


