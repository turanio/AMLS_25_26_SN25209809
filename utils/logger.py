import logging
import os
from logging.handlers import RotatingFileHandler


class CustomLogger:
    LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def _create_handler(self, log_file: str, level: int) -> RotatingFileHandler:
        """
        Create a rotating file handler with standard formatting.
        """
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        handler.setLevel(level)

        formatter = logging.Formatter(self.LOG_FORMAT, self.DATE_FORMAT)
        handler.setFormatter(formatter)
        return handler

    def get_logger(self, name: str, log_file: str) -> logging.Logger:
        """
        Create or retrieve a configured logger.

        Args:
            name (str): Logger name (e.g. ModelA.Train)
            log_file (str): Path to main log file

        Returns:
            logging.Logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.hasHandlers():
            return logger

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(self.LOG_FORMAT, self.DATE_FORMAT)
        )

        info_handler = self._create_handler(log_file, logging.INFO)
        warning_handler = self._create_handler("logs/warnings.log", logging.WARNING)
        error_handler = self._create_handler("logs/errors.log", logging.ERROR)

        logger.addHandler(console_handler)
        logger.addHandler(info_handler)
        logger.addHandler(warning_handler)
        logger.addHandler(error_handler)

        logger.propagate = False
        return logger
