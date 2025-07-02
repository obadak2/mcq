# src/logger.py
import logging

def setup_logger(log_level: str = "INFO"):
    logging.basicConfig(
        level=log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    return logger
