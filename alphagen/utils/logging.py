import logging
import os
from typing import Optional


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    logger = logging.getLogger(name)
    while logger.hasHandlers():
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")

    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_null_logger() -> logging.Logger:
    logger = logging.getLogger("null_logger")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger
