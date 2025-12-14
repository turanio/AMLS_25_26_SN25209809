
import logging
import os
from logging.handlers import RotatingFileHandler


class CustomLogger:
    LOG_FORMAT = (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def _create_handler(self, log_file: str, level: logging):
        """Creating handler file with specified structure of log and date format.
        
        Args:
            log_file (str): Name of the file.
            level (logging): Level of logging. Example as: Info, Warnning, Debug.
        
        Returns:
            RotatingFileHandler: File Handler with specified structure of log and date format
        """
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        handler.setLevel(level)
        formatter = logging.Formatter(CustomLogger.LOG_FORMAT, CustomLogger.DATE_FORMAT)
        handler.setFormatter(formatter)
        return handler

    def get_logger(self, name: str, log_file: str):
        """
        Create or retrieve a configured logger.

        Args:
            name (str): Name of the file.
            log_file (str): Directory to the log file.

        Returns:
            Logger: Logging object to use for logging the data.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(CustomLogger.LOG_FORMAT, CustomLogger.DATE_FORMAT))

        file_handler = self._create_handler(log_file, logging.INFO)
        warning_handler = self._create_handler(
            "logs/warnings.log", logging.WARNING
        )
        error_handler = self._create_handler(
            "logs/errors.log", logging.ERROR
        )

        logger.addHandler(console)
        logger.addHandler(file_handler)
        logger.addHandler(warning_handler)
        logger.addHandler(error_handler)

        logger.propagate = False
        return logger
