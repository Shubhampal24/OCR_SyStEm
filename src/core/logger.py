import logging
import sys
from src.core.config import settings

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a standardized logger for the application.
    In an industrial app, you never use print(). You use a logger to track exactly 
    what time an event happened and from which file.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        # If DEBUG_MODE is true in .env, show all logs. Otherwise, only show INFO and above.
        log_level = logging.DEBUG if settings.DEBUG_MODE else logging.INFO
        logger.setLevel(log_level)
        
        # Output logs to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Standardized format: "2026-06-24 10:00:00 - module_name - INFO - The message"
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger
