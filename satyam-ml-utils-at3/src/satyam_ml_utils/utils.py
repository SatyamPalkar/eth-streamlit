import os
import random
import numpy as np
import time
import logging


def seed_everything(seed: int = 42):
    """Ensure reproducibility across numpy, random, and PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global seed set to {seed}")


def setup_logger(name="ml_utils", log_file="ml_utils.log", level=logging.INFO):
    """Setup a logger that writes to console + file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name="Timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        print(f"⏱️ {self.name} finished in {duration:.2f} seconds")
