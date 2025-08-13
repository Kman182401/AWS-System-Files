import logging

# Set up a basic logger
logger = logging.getLogger("trade_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_trade_results(message):
    """
    Logs trade results or audit messages.
    """
    logger.info(message)
