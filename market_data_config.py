# IBKR TWS live market data config (Level 2 ready, robust, and efficient)

# Max daily loss as a percentage of starting equity (for learning/paper trading)
MAX_DAILY_LOSS_PCT = 0.02  # 2%

# Max number of trades allowed per day (for learning/paper trading)
MAX_TRADES_PER_DAY = 20

# Max position exposure (absolute value, for learning/paper trading)
MAX_POSITION_EXPOSURE = 3  # e.g., never hold more than 3 contracts long or short

# Max order size (contracts per order, for learning/paper trading)
MAX_ORDER_SIZE = 2  # e.g., never send an order for more than 2 contracts

# TWS connection settings
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 4002 # 4001 for live trading, 4002 for paper trading
IBKR_CLIENT_ID = 101 # Unique, non-zero client ID

# Connection/retry settings
IBKR_RETRY_INTERVAL = 5   # seconds between reconnect attempts
IBKR_TIMEOUT = 30         # seconds for API call timeouts

# Market data type
IBKR_MARKET_DATA_TYPE = "REALTIME"  # Options: REALTIME, DELAYED, FROZEN

# Logging and monitoring
IBKR_LOGGING_ENABLED = True
IBKR_LOG_FILE = "ibkr_market_data.log"
IBKR_LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# Dynamic subscription management
IBKR_DYNAMIC_SUBSCRIPTION = True  # Only subscribe to active symbols

# Resource monitoring (optional, for scaling)
IBKR_RESOURCE_MONITORING = True

# Main loop poll interval (optional, for pipeline sleep)
IBKR_POLL_INTERVAL = 1  # seconds

# Symbols to track (must match adapter mapping)
IBKR_SYMBOLS = [
    'ES1!',    # S&P 500 E-mini (continuous via front-month selection)
    'NQ1!',    # Nasdaq 100 E-mini (continuous via front-month selection)
    'GBPUSD',  # proxy via 6B (CME British Pound futures)
    'EURUSD',  # proxy via 6E (CME Euro futures)
    'AUDUSD',  # proxy via 6A (CME Aussie Dollar futures)
    'XAUUSD',  # proxy via GC (COMEX Gold futures)
]
