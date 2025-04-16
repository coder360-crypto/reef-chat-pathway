import os
from dotenv import load_dotenv

import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class EmbedderConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self._load_config()
            self.initialized = True

    def _load_config(self):
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        self.PORT = int(os.getenv("PORT", "8007"))
        logger.info("Config Loaded:")
        logger.info("PORT: %s", self.PORT)
        logger.info("-" * 55)


if __name__ == "__main__":
    config = EmbedderConfig()
