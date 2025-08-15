import logging
import sys


def setup_logger(name: str = "translatotron"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] - %(name)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


logger = setup_logger()
