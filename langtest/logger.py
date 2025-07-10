import os
import logging
from logging.handlers import RotatingFileHandler


class Logger:
    """
    A class to configure a logger with a File Handler.

    """

    def __init__(
        self,
        name="langtest",
        log_dir=None,
        level=logging.INFO,
        max_bytes=10485760,
        backup_count=5,
    ):
        """
        Initialize and configure a logger using a File Handler.

        Args:
            name (str): The name of the logger.
            log_dir (str): The directory to store the log files.
            level (int): The log level.
            max_bytes (int): The maximum size of the log file in bytes. default is 10MB.
            backup_count (int): The number of backup log files to keep.

        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if log_dir is None:
            log_dir = os.path.join(os.path.expanduser("~"), ".langtest/logs/")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "langtest.log")

        # Create a rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)

        # Create and set the formatter
        formatter = logging.Formatter(
            "%(asctime)-20s | %(levelname)-8s | %(name)-10s -> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Return the logger object.

        Returns:
            logging.Logger: The logger object.

        """
        return self.logger

    def set_level(self, level):
        """
        Set the log level of the logger.

        Args:
            level (int): The log level.

        """
        self.logger.setLevel(level)

    def add_console_handler(self, level=logging.INFO):
        """
        Add a console handler to the logger.

        Args:
            level (int): The log level.

        """
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)-20s | %(levelname)-8s | %(name)-10s -> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def remove_console_handler(self):
        """
        Remove the console handler from the logger.

        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, logging.StreamHandler):
                self.logger.removeHandler(handler)

    def remove_file_handler(self):
        """
        Remove the file handler from the logger.

        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            if isinstance(handler, RotatingFileHandler):
                self.logger.removeHandler(handler)


# Create a logger object
logger = Logger().get_logger()
