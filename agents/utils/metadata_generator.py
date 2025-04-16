# Import system and path handling
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standard libraries
import ast
import hashlib
import json
import logging
from typing import List, Optional, Tuple

# Import external dependencies
import requests
from config import AgentsConfig as Config
from diskcache import Cache
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Initialize environment and logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prompt template for entity selection
choose_entities_prompt = """
You are an expert at choosing relevant key entities according to the query. \
Given the query and key entities from a source, choose the most relevant key entities based on the query.
It is not necessary for the related entity to be explicitly mentioned in the query.

<query>
{query}
</query>

<entities>
{entities}
</entities>

**Guidelines**
1. Return only the MOST relevant key entities as a Python list of strings
2. Do not include any additional information in the output
3. If there are multiple similar entities, include all of them
4. Limit response to maximum 20 most relevant entities

Return only a valid Python list of strings.
Example Output: ["entity1", "entity2", "entity3"]
"""

# Prompt template for topic selection
choose_topics_prompt = """
You are an expert at choosing relevant topics according to the query. \
Given the query and the topics extracted from the documents, choose the most relevant topics based on the query.

<query>
{query}
</query>

<topics>
{topics}
</topics>

**Guidelines**
1. Return only the MOST relevant topics as a Pythonlist of strings.
2. Do not include any additional information in the output.
3. Do not give more than 10 topics.

Example input:
query: "What are the key terms and conditions of the Residential Lease Agreement between Emily E Roberts and Michael Thompson?"
topics: [exhaustive list of topics]
Output:
["lease agreement", "residential lease", "terms and conditions", "emily e roberts", "michael thompson"]

Example output:
["lease agreement", "residential lease", "key terms"]

Return only the Python list of strings in the specified format.
"""


class MetadataGenerator:
    """
    This class is used to get filter on a query on a user level.

    Args:
        jwt_token (Optional[str]): Authentication token for API requests
        selected_docs (Optional[List[str]]): List of document IDs to filter by
        llm (BaseChatModel): Language model instance for generating responses
    """

    _cache = Cache(
        os.path.join(os.path.dirname(__file__), ".metadata-cache")
    )  # stores user based metadata cache
    CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        jwt_token: Optional[str] = None,
        selected_docs: Optional[List[str]] = [],
        llm: BaseChatModel = ChatOpenAI(
            api_key=Config().OPENAI_API_KEY, model="gpt-4o"
        ),
    ):
        config = Config()
        self.inputs_url = config.INPUTS_URL
        self.jwt_token = jwt_token
        self.all_metadatas = []  # all metadatas available for a user
        self.metadatas = []  # selected docs based metadatas
        self.selected_docs = selected_docs  # selected docs per query
        self.key_entities = set()  # key entitites filted by selected docs
        self.client = llm
        self.model = llm.model_name
        self.folder_ids = []
        self._update_metadatas()

    def _get_cache_key(self) -> str:
        """
        Generate a unique cache key based on JWT token.
        
        Returns:
            str: Hashed cache key
        """
        token = self.jwt_token or "default"
        return f"metadata_{hashlib.sha256(token.encode()).hexdigest()}"

    def _get_entity_docs(self, entities: list[str]) -> list[str]:
        """
        Get the document ids with any overlapping entities.
        """
        return list(
            {
                document["id"]
                for document in self.metadatas
                if any(entity in document["key_entities"] for entity in entities)
            }
        )

    def _update_metadatas(self) -> None:
        """Fetch and cache metadata from the server."""
        cache_key = self._get_cache_key()
        cached_data = self._cache.get(cache_key)

        if cached_data is not None:
            self.all_metadatas = cached_data["all_metadatas"]
            logger.info("Using cached metadatas (all)")

        else:

            try:
                doc_inputs = requests.get(
                    self.inputs_url, headers={"Authorization": self.jwt_token}
                )
                data = doc_inputs.json()
                self.all_metadatas = data["metadatas"]

                cache_data = {
                    "all_metadatas": self.all_metadatas,
                }
                self._cache.set(cache_key, cache_data, expire=self.CACHE_TTL)

            except Exception as e:
                logger.error(f"Error in updating metadatas: {str(e)}", exc_info=True)

        if not self.selected_docs:
            self.metadatas = self.all_metadatas
        else:
            self.metadatas = [
                metadata
                for metadata in self.all_metadatas
                if metadata["id"] in self.selected_docs
            ]
        self.key_entities = [
            entity for metadata in self.metadatas for entity in metadata["key_entities"]
        ]
        self.folder_ids = set([metadata["parents"][0] for metadata in self.metadatas])
        logger.info("Updated metadata and cache")

    def _get_entities(self, query: str) -> List[str]:
        """Get relevant entities based on the query using LLM."""
        content = choose_entities_prompt.format(
            query=query, entities=json.dumps(list(self.key_entities))
        )
        if "gpt" not in self.model:
            content = content[:4000]
        response = self.client.invoke(
            input=[{"role": "user", "content": content}],
            max_tokens=1000,
            temperature=0.0,
        )

        try:
            result_string = response.content.lower()
            result = eval(
                result_string.replace("```python", "").replace("```", "").strip()
            )
            if isinstance(result, list):
                return result
        except Exception as e:
            logger.error(f"Error parsing LLM metadata response with eval: {str(e)}")
            logger.error(f"Raw response: {response.content}")
            try:
                result = ast.literal_eval(response.content)
                if isinstance(result, list):
                    return result
            except Exception as e:
                logger.error(
                    f"Error parsing LLM metadata response with ast.literal_eval: {str(e)}"
                )
                return []

    def _get_topics(self, doc_ids: list[str]) -> list[str]:
        """
        Get the topics for the documents.
        """
        return list(
            set(
                {
                    topic
                    for document in self.metadatas
                    if document["id"] in doc_ids
                    for sub_doc_id in document["topics"]
                    for topic in document["topics"][sub_doc_id]
                }
            )
        )

    def _get_relevant_topics(self, query, topics: list[str]) -> list[str]:
        """
        Get the relevant topics based on the query using an LLM Call.
        """

        content = choose_topics_prompt.format(query=query, topics=str(topics))
        if "gpt" not in self.model:
            content = content[:4000]
        response = self.client.invoke(
            input=[{"role": "user", "content": content}],
            max_tokens=1000,
            temperature=0.0,
        )

        try:
            result_string = response.content.lower()
            result = eval(
                result_string.replace("```python", "").replace("```", "").strip()
            )

            logger.info(f"Relevant topics found: {result}")
            if isinstance(result, list):
                return result
        except Exception as e:
            logger.error(f"Error parsing LLM metadata response with eval: {str(e)}")
            logger.error(f"Raw response: {response.content}")
            try:
                result = ast.literal_eval(response.content)
                if isinstance(result, list):
                    return result
            except Exception as e:
                logger.error(
                    f"Error parsing LLM metadata response with ast.literal_eval: {str(e)}"
                )
                return []

    def _get_relevant_docs_and_entities(
        self, query: str
    ) -> Tuple[dict[str, List[str]], List[str]]:
        """
        Get relevant document names and their key entities based on the query.

        Returns:
        - relevant_docs: dict = {"file1": [key_entitites], ...}
        - relevant_entities: list
        """
        logger.info(f"Getting relevant docs for: {query}")

        relevant_entities = self._get_entities(query)

        logger.info(f"Relevant entities found: {str(relevant_entities)}")

        relevant_docs = {
            document["name"]: [
                entity
                for entity in document["key_entities"]
                if entity in relevant_entities
            ]
            for document in self.metadatas
            if any(entity in document["key_entities"] for entity in relevant_entities)
        }

        logger.info(f"Relevant docs found: {str(relevant_docs)}")
        return relevant_docs, relevant_entities

    def _merge_and(self, filter1: str, filter2: str):
        if not filter1 and not filter2:
            return ""
        if not filter1:
            return filter2
        if not filter2:
            return filter1
        return f"({filter1}) && ({filter2})"

    def get_filter_and_relevant_docs(
        self,
        query: str,
        useTopic: bool = True,
        domain: Optional[str] = "finance",
    ) -> Tuple[str, dict[str, List[str]]]:
        """
        Get the filter for the query based on the metadata.
        - Docs are chosen based on key entities.
        - Sub docs are chosen based on topics found in the relevant docs.
        - Search space is limited to selected docs if chosen by user

        Args:
            query (str): The search query
            useTopic (bool): Whether to include topic-based filtering
            domain (Optional[str]): Domain context for the search

        Returns:
            Tuple[str, dict[str, List[str]]]: Filter string and relevant documents with their entities
        """

        logging.info(f"Getting metadata filter for: {query}")
        relevant_docs, relevant_entities = self._get_relevant_docs_and_entities(query)
        doc_ids = self._get_entity_docs(relevant_entities)
        logger.info(f"Doc ids found: {str(doc_ids)}")
        if doc_ids:
            entity_filter = " || ".join([f"id == `{doc_id}`" for doc_id in doc_ids])
        else:
            entity_filter = ""

        topic_filter = ""

        if useTopic:
            topics = self._get_topics(doc_ids)
            relevant_topics = self._get_relevant_topics(query, topics)
            if relevant_topics:
                topic_filter = " || ".join(
                    [f"contains(topics, `{topic}`)" for topic in relevant_topics]
                )

        filter = self._merge_and(entity_filter, topic_filter)

        folder_filter = " || ".join(
            [f"contains(parents,`{folder_id}`)" for folder_id in self.folder_ids]
        )

        filter = self._merge_and(filter, folder_filter)

        logger.info(f"Filter found: {filter}")
        return filter, relevant_docs

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the entire cache."""
        cls._cache.clear()


if __name__ == "__main__":

    metadata_generator = MetadataGenerator(jwt_token=Config().JWT_TOKEN)
    query = "What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why."
    filter, relevant_docs = metadata_generator.get_filter_and_relevant_docs(query)
    print(f"Filter: ", filter)
    print("Relevant docs: ", relevant_docs)
