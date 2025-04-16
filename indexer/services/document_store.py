# Import system libraries
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standard libraries
import json
import logging

# Import third-party libraries
import pathway as pw
import requests
from config import IndexerConfig as Config
from pathway.stdlib.indexing.bm25 import TantivyBM25Factory
from pathway.stdlib.indexing.hybrid_index import HybridIndexFactory
from pathway.stdlib.indexing.nearest_neighbors import UsearchKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer
from pydantic import InstanceOf
from utils.custom_splitter import ContextualMetadataSplitter
from utils.embedding_client import EmbeddingClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DocumentStoreServerWrapper:
    """
    A wrapper class for managing document store server operations including initialization,
    creation, and running of the server.

    Attributes:
        host (str): Host address for the vector database
        port (int): Port number for the vector database
        server: Document store server instance
        client: Client instance
        with_cache (bool): Flag to enable/disable caching
        cache_backend: Filesystem cache backend instance
        terminate_on_error (bool): Flag to control error handling behavior
        embedder: Embedding client instance
        splitter: Document splitter instance
    """

    def __init__(self):
        # Initialize configuration
        config = Config()

        self.host = config.VECTOR_DB_HOST
        self.port = config.VECTOR_DB_PORT
        self.server = None
        self.client = None
        self.with_cache: bool = True

        # Set up cache backend
        self.cache_backend: InstanceOf[pw.persistence.Backend] = (
            pw.persistence.Backend.filesystem(".cache/embedding-cache")
        )
        self.terminate_on_error: bool = False
        self.embedder = EmbeddingClient(cache_strategy=pw.udfs.DefaultCache())
        self.splitter = ContextualMetadataSplitter(chunk_overlap=400, chunk_size=4000)

    def create_server(
        self,
        data,
        embedder=None,
        splitter=None,
    ):
        """
        Creates a document store server with specified configuration.

        Args:
            data: Input data for document store
            embedder: Custom embedder instance (optional)
            splitter: Custom splitter instance (optional)

        Raises:
            Exception: If server creation fails at any step
        """
        # Use default embedder if none provided
        if embedder is None:
            embedder = self.embedder
        if splitter is None:
            splitter = self.splitter

        try:
            # Initialize UsearchKnnFactory for vector search
            self.usearch_knn_factory = UsearchKnnFactory(
                dimensions=embedder.get_embedding_dimension(),
                reserved_space=1000,
                connectivity=0,
                expansion_add=0,
                expansion_search=0,
                embedder=embedder,
            )
        except Exception as e:
            logger.error("Error creating UsearchKnnFactory: %s", str(e))
            raise

        try:
            # Initialize BM25 factory for text search
            self.bm25_factory = TantivyBM25Factory()
        except Exception as e:
            logger.error("Error creating BM25Factory: %s", str(e))
            raise

        self.retriever_factories = [self.bm25_factory, self.usearch_knn_factory]

        try:
            # Create hybrid index and document store
            self.hybrid_index_factory = HybridIndexFactory(
                retriever_factories=self.retriever_factories
            )
            self.document_store = DocumentStore(
                *data,
                retriever_factory=self.hybrid_index_factory,
                splitter=splitter,
            )
        except Exception as e:
            logger.error("Error creating DocumentStore Instance: %s", str(e))
            raise

        try:
            # Initialize the server
            self.server = DocumentStoreServer(
                host=self.host, port=self.port, document_store=self.document_store
            )
            logger.info("Server created successfully")
        except Exception as e:
            logger.error("Error creating server: %s", str(e))
            raise

    def run_server(self, with_cache=True, threaded=False):
        """
        Starts the document store server.

        Args:
            with_cache (bool): Enable/disable caching
            threaded (bool): Run server in threaded mode

        Raises:
            Exception: If server fails to start
        """
        try:
            logger.info("Starting server on %s:%s", self.host, self.port)
            self.server.run(
                with_cache=with_cache,
                threaded=threaded,
                cache_backend=self.cache_backend,
                terminate_on_error=self.terminate_on_error,
            )
            logger.info("Server started successfully")
        except Exception as e:
            logger.error("Error running server: %s", str(e))
            raise
