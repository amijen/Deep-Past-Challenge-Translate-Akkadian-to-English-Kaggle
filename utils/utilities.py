import logging
import torch
import numpy as np
import random
import gc

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(log_level)
    return logger


def seed_everything(seed: int, logger: logging.Logger):
    """Sets random seeds for reproducibility across libraries."""
    logger.info(f"Setting global seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clear_memory():
    """Aggressively clears GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()