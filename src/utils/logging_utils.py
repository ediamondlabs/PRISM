import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_config(logger: logging.Logger, config: dict) -> None:
    """Log configuration parameters in a structured format.

    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("=" * 80)
    _log_dict_recursive(logger, config, indent=0)
    logger.info("=" * 80)


def _log_dict_recursive(logger: logging.Logger, d: dict, indent: int = 0) -> None:
    """Recursively log dictionary contents with indentation.

    Args:
        logger: Logger instance
        d: Dictionary to log
        indent: Current indentation level
    """
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info("  " * indent + f"{key}:")
            _log_dict_recursive(logger, value, indent + 1)
        else:
            logger.info("  " * indent + f"{key}: {value}")
