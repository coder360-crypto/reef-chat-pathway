# Import system modules
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import logging
import logging

# Import external dependencies
import cohere
from config import AgentsConfig as Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CohereReranker:
    """
    A class to rerank search results using Cohere's reranking model.
    """
    def __init__(self):
        """
        Initializes CohereReranker with document store client and API key
        """
        config = Config()
        self.api_key = config.COHERE_API_KEY

    def rerank(self, query, result, top_k=3):
        """
        Reranks search results using Cohere's reranking model

        Args:
            query (str): User's search query text
            result (list): List of documents to rerank
            top_k (int): Number of top results to return

        Returns:
            list: List of top_k documents, sorted by relevance score
        """
        try:
            if result is None:
                logger.error("Error: Failed to retrieve results from document store")
                return []

            retrieved_context = result

            try:
                # Initialize Cohere client
                co = cohere.Client(self.api_key)
                reranked = co.rerank(
                    query=query,
                    documents=retrieved_context,
                    model="rerank-english-v2.0",
                    top_n=top_k,
                )
            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}")
                return []

            # Process reranked results
            reranked_documents = []
            for idx, result in enumerate(reranked.results):
                # reranked_documents.append((result.relevance_score, retrieved_context[result.index]))
                reranked_documents.append(retrieved_context[result.index])

            return reranked_documents[:top_k]
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return []


# Main execution block
if __name__ == "__main__":
    cohere_reranker = CohereReranker()
    sample_query = "What is machine learning?"

    # Test Cohere reranking
    logger.info("\nCohere reranking results:")
    cohere_results = cohere_reranker.rerank(sample_query, top_k=3)
    for score, doc in cohere_results:
        logger.info(f"\nScore: {score:.4f}")
        logger.info(f"Document: {doc[:200]}...")
