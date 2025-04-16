import logging
import os
from typing import Dict

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class AgentsConfig:

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
        # Misc details for the agent service
        self.PORT = int(os.getenv("PORT", "8006"))
        self.GUARDRAILS_API_URL = os.getenv(
            "GUARDRAILS_API_URL", "http://localhost:8008/validate"
        )

        # LLM API details
        self.GROQ_API_BASE = "https://api.groq.com/openai/v1"
        self.OPENAI_API_BASE = "https://api.openai.com/v1"
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")
        self.RETRIEVER_API_URL = os.getenv(
            "RETRIEVER_API_URL", "http://localhost:8666/v1/retrieve"
        )
        self.INPUTS_URL = os.getenv(
            "INPUTS_URL", "http://localhost:8000/api/get-user-doc-metadata"
        )
        self.JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
        self.JWT_TOKEN = os.getenv(
            "JWT_TOKEN",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRhbnVlZHUxMjhAZ21haWwuY29tIn0.TjzYiFh4zBdmRwmf_r0duBfu1jeT1xgSjE3YY2WOdIo",
        )

        # Agent API keys
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
        self.POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
        self.FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
        self.FINHUB_API_KEY = os.getenv("FINHUB_API_KEY")
        self.COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.JINA_API_KEY = os.getenv("JINA_API_KEY")
        self.FINHUB_API_KEY = os.getenv("FINHUB_API_KEY")
        self.COURTLISTENER_API_KEY = os.getenv("COURTLISTENER_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.FMP_API_KEY = os.getenv("FMP_API_KEY")
        self.CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")
        self.SERP_API_KEY = os.getenv("SERP_API_KEY")
        self.RAPID_API_KEY = os.getenv("RAPID_API_KEY")
        self.IK_API_KEY = os.getenv("IK_API_KEY")
        self.US_LAWS_FOLDER_ID = os.getenv("US_LAWS_FOLDER_ID")

        logger.info(f"Config loaded: {self.__dict__}")
