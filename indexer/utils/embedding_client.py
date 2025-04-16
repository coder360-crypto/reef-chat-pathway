# Import required libraries
import aiohttp
import numpy as np
from config import IndexerConfig as Config
from pathway.internals import udfs
from pathway.xpacks.llm.embedders import BaseEmbedder
import logging

# Set up logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class EmbeddingClient(BaseEmbedder):
    """Makes calls to a custom fastapi endpoint to generate embeddings.

    This class implements the BaseEmbedder interface to provide embeddings by making HTTP requests
    to a FastAPI endpoint. The endpoint should accept POST requests with JSON payload containing
    an "input" field with the text to embed, and return JSON with an "embedding" field containing
    the vector representation.

    The embedding endpoint URL is configured via Config.EMBEDDING_API_URL.

    Args:
        capacity (int | None): Maximum number of concurrent requests. None means unlimited.
        retry_strategy (udfs.AsyncRetryStrategy | None): Strategy for retrying failed requests.
        cache_strategy (udfs.CacheStrategy | None): Strategy for caching embedding results.
        model (str): The model to use for embedding.

    Methods:
        __wrapped__(input: str, **kwargs) -> np.ndarray:
            Makes the HTTP request to get embeddings for the input text.
            Returns a numpy array containing the embedding vector.
    """

    def __init__(
        self,
        *,
        capacity: int = 10,
        max_retries: int = 5,
        cache_strategy: udfs.CacheStrategy = None,
    ):
        # Create retry strategy with exponential backoff
        retry_strategy = udfs.ExponentialBackoffRetryStrategy(max_retries=max_retries)
        # Set up async executor with retry and timeout settings
        executor = udfs.async_executor(
            capacity=capacity,
            retry_strategy=retry_strategy,
            timeout=20.0,
        )
        # Initialize parent class
        super().__init__(
            executor=executor,
            cache_strategy=cache_strategy,
        )
        # Get config and set API URL
        config = Config()
        self.url = config.EMBEDDING_API_URL

    async def __wrapped__(self, input: str, **kwargs) -> np.ndarray:
        # Create HTTP session and make POST request to embedding API
        async with aiohttp.ClientSession() as session:
            response = await session.post(self.url, json={"text": input})
            data = await response.json()
            # Log first 5 elements of embedding for debugging
            logger.info(data["embedding"][:5])
            # Return embedding as numpy array
            return np.array(data["embedding"])
