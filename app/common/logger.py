import os 
import logging

from datetime import datetime

LOG_D = "running_d"
os.makedirs(LOG_D, exist_ok=True)

LOG_FILE = os.path.join(LOG_D, f"{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    encoding="utf-8"
)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
