import logging
import sys, os
from functools import lru_cache

def setup_logger(log_level:str="INFO")->None:
    # Create formatter
    formatter=logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%s %H:%M:%S",
    )
    
    # Configure root logger
    root_logger=logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.Handlers[:]:
        root_logger.removeHandler(handler)
        
    console_handler=logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
@lru_cache
def get_logger(name:str)->logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

class Logger_Mixin:
    """Mixin class to add logging capability to classes."""
    def logger(self)->logging.Logger:
        return get_logger(self.__class__.__name__)