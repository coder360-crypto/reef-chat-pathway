import logging
import os
import sys

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pathway as pw
from config import IndexerConfig as Config
from services.document_store import DocumentStoreServerWrapper
from services.drive_connector import DriveConnector


class Indexer:
    """
    A service that indexes documents from various connectors into a vector database.
    
    Args:
        config (Config): Configuration object containing indexer settings
    """

    def __init__(self, config: Config):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.config = config

        self.logger.info("Initializing Indexer service")
        # Initialize list of document connectors
        self.connectors = [
            DriveConnector(folder_id=config.GDRIVE_LINK),

        ]

        self.docs = []

        self.index_documents()

    def index_documents(self):
        """
        Fetches documents from all connectors and initiates indexing process.
        """
        self.logger.info("Starting document indexing process")
        # Get parsed documents from each connector
        for connector in self.connectors:
            self.docs.append(connector.get_parsed_table())
        self._add_to_vector_db()

    def _add_to_vector_db(self):
        """
        Initializes vector database connection and adds documents to it.
        """
        self.logger.info("Initializing vector database connection")
        self.vector_store_server = DocumentStoreServerWrapper()

        self.logger.info("Creating vector database server")

        self.vector_store_server.create_server(self.docs)

        self.logger.info("Starting vector database server")
        self.vector_store_server.run_server()


# Entry point of the script
if __name__ == "__main__":
    config = Config()
    indexer = Indexer(config)
    pw.run_all()
