"""levelapp/aspects/logger.py"""
import logging
import sys
from typing import Optional


class Logger:
    """Centralized logger for Levelapp (AOP-style)."""

    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(
            cls,
            name: str = "levelapp",
            level: int = logging.INFO,
            log_to_file: Optional[str] = None,
    ) -> logging.Logger:
        """
        Get a configured logger instance (singleton).

        Args:
            name (str): Logger name.
            level (int): Logging level (default=INFO).
            log_to_file (Optional[str]): File path for logging (default=None).

        Returns:
            logging.Logger: Configured logger instance.
        """
        if cls._instance:
            return cls._instance

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False  # prevent double logging

        # Formatter with contextual info
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger_.addHandler(console_handler)

        # Optional file handler
        if log_to_file:
            file_handler = logging.FileHandler(log_to_file)
            file_handler.setFormatter(formatter)
            logger_.addHandler(file_handler)

        cls._instance = logger_
        return logger_


# Global access
logger = Logger.get_logger()
