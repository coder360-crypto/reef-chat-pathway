import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class InterfaceConfig:
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
        self.PORT = int(os.getenv("INTERFACE_PORT", "8001"))
        self.AGENTS_API_URL = os.getenv("AGENTS_API_URL", "http://localhost:8006")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DB_URL = os.getenv("DB_URL")
        logger.info("Interface config loaded successfully")
        logger.info("AGENTS_API_URL: %s", self.AGENTS_API_URL)
