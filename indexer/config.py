import os

from dotenv import load_dotenv

import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class IndexerConfig:
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
        logger.info("Config Loaded:")
        self.GDRIVE_LINK = os.getenv("GDRIVE_LINK", "")

        self.EMBEDDING_API_URL = os.getenv(
            "EMBEDDING_API_URL", "http://localhost:8007/embeddings"
        )
        self.VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "0.0.0.0")
        self.VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8666"))
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # For testing purposes use this secret key
        self.JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
        self.UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")
        self.UNSTRUCTURED_SERVER_URL = os.getenv("UNSTRUCTURED_SERVER_URL")
        self.DB_URL = os.getenv("DB_URL")

        logger.info("GDRIVE_LINK: %s", self.GDRIVE_LINK)

        logger.info("ANTHROPIC_API_KEY: %s", self.ANTHROPIC_API_KEY)
        logger.info("EMBEDDING_API_URL: %s", self.EMBEDDING_API_URL)
        logger.info("VECTOR_DB_HOST: %s", self.VECTOR_DB_HOST)
        logger.info("VECTOR_DB_PORT: %s", self.VECTOR_DB_PORT)
        logger.info("DB_URL: %s", self.DB_URL)

        logger.info("-" * 55)
