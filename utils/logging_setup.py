import logging

def setup_logging() -> logging.Logger:
    """
    Set up logging configuration.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

logger = setup_logging()