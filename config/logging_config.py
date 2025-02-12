import logging
from typing import Dict, Any

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Sets up logging configuration based on the provided config.
    
    Args:
        config: Configuration dictionary containing logging settings
    """
    logging_config = config.get("logging", {})
    level_str = logging_config.get("level", "INFO").upper()
    format_str = logging_config.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        level = getattr(logging, level_str)
    except AttributeError:
        level = logging.INFO
        logging.warning(
            f"Invalid logging level in config.yaml: {level_str}. Using INFO instead."
        )

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str
    )

    # Set level for specific loggers
    loggers = logging_config.get("loggers", {})
    for logger_name, logger_level in loggers.items():
        try:
            logger = logging.getLogger(logger_name)
            logger.setLevel(getattr(logging, logger_level.upper()))
        except AttributeError:
            logging.warning(
                f"Invalid logging level for {logger_name}: {logger_level}"
            )