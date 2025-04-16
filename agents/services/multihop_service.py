import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import requests
from config import AgentsConfig as Config
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from utils.reranker import CohereReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()


@dataclass
class Document:
    content: str
    id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class RetrievalHop:
    query: str
    retrieved_documents: List[Document]
    analysis: Dict
    refined_query: Optional[str] = None


class HopAnalysis(BaseModel):
    novelty_analysis: List[str] = Field(
        description="New relevant information found in current chunks"
    )
    deficiency_analysis: List[str] = Field(
        description="Information still missing based on query intent"
    )
    refined_query: str = Field(description="Refined query for next hop")


def get_api_details_from_llm(llm: str):
    logger.info(f"Getting API details for LLM: {llm}")
    if "gpt" in llm or "o1" in llm:
        logger.info("Using OpenAI API configuration")
        return "OPENAI_API_KEY", config.OPENAI_API_BASE
    elif "llama" in llm:
        logger.info("Using Groq API configuration")
        return "GROQ_API_KEY", config.GROQ_API_BASE
    else:
        logger.warning(f"Unknown LLM type: {llm}, defaulting to OpenAI")
        return "OPENAI_API_KEY", config.OPENAI_API_BASE


class MultiHopRetriever:
    def __init__(self):
        """Initialize the MultiHopRetriever with API keys and services."""
        self.llm = None
        self.retriever_url = config.RETRIEVER_API_URL
        self.retriever = requests.Session()
        self.domain = "finance"

        self._init_prompt()

        logger.info("MultiHopRetriever initialized successfully")

    def _setup_llm(self, llm_name: str):
        """Set up the LLM client based on the provided LLM name."""
        if self.llm and self.llm.model_name == llm_name:
            logger.info(f"LLM already configured for {llm_name}")
            return

        logger.info(f"Configuring LLM: {llm_name}")
        api_key_name, model_url = get_api_details_from_llm(llm_name)

        try:
            api_key = os.environ.get(api_key_name)
            if not api_key:
                raise ValueError(f"Missing API key: {api_key_name}")

            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name=llm_name,
                temperature=0.4,
                base_url=model_url,
            )
            logger.info(f"LLM configured successfully for {llm_name}")

        except Exception as e:
            logger.error(f"Error in setting up LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")

    def _init_prompt(self):
        """Initialize the unified analysis prompt."""
        self.analysis_prompt = ChatPromptTemplate.from_template(
            """
        Analyze the information retrieved so far and provide guidance for the next step.
        
        ORIGINAL QUERY: {original_query}
        
        PREVIOUS CHUNKS RETRIEVED:
        {previous_chunks}
        
        NEW CHUNKS JUST RETRIEVED IN THIS HOP:
        {new_chunks}
        
        Task: Analyze all information above and output a JSON object with the following structure:
        {{
            "novelty_analysis": [
                "List specific new facts/insights found ONLY in the new chunks that are relevant to the original query",
                "Focus on information that wasn't covered in previous chunks"
            ],
            "deficiency_analysis": [
                "List specific aspects of the original query that still need more information",
                "Identify important gaps in our current understanding based on the query's intent and content"
            ],
            "refined_query": "Write a focused search query that specifically targets the gaps identified in deficiency_analysis while avoiding topics well-covered in novelty_analysis"
        }}

        REQUIREMENTS:
        1. novelty_analysis: Only include new information from the latest chunks
        2. deficiency_analysis: Be specific about what's missing in the query according to chunks retrieved till now
        3. refined_query: Target the identified gaps while ensuring to avoid redundant information

        OUTPUT ONLY THE JSON OBJECT, NO OTHER TEXT.
        """
        )

    def _retrieve_documents(
        self, query: str, k: int = 3, metadata_filter: str = ""
    ) -> List[Document]:
        """
        Retrieve documents from the retriever endpoint.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            List of Document objects
        """
        retriever_url = config.RETRIEVER_API_URL

        try:
            metadata_filter = (
                metadata_filter
                if (metadata_filter and len(metadata_filter) > 0)
                else None
            )
            logger.info(retriever_url)
            response = self.retriever.post(
                retriever_url,
                json={"query": query, "k": k, "metadata_filter": metadata_filter},
            )

            response.raise_for_status()
            results = response.json()

            # Convert results to Document objects
            documents = []
            for i, result in enumerate(results[:k]):
                doc = Document(
                    content=result["text"],
                    metadata=result.get("metadata"),
                )
                documents.append(doc)

            logger.info(
                f"Retrieved {len(documents)} documents for query: {query[:50]}..."
            )
            return documents

        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            return []

    def _get_novel_documents(
        self,
        query: str,
        current_doc_strings: Set[str],
        k: int,
        metadata_filter: str = "",
    ) -> List[Document]:
        """
        Retrieve documents ensuring at least min_new new documents.

        Args:
            query: Search query
            seen_ids: Previously seen document IDs
            k: Total documents to retrieve
            min_new: Minimum number of new documents required
        """
        try:
            # Retrieve initial batch of documents
            potential_new_docs = self._retrieve_documents(
                query, k=k, metadata_filter=metadata_filter
            )

            new_cnt = 0
            novel_docs = []
            for doc in potential_new_docs:
                if doc.content not in current_doc_strings:
                    new_cnt += 1
                    current_doc_strings.add(doc.content)
                    novel_docs.append(doc)

            logger.info(f"Retrieved {len(novel_docs)} novel documents")
            return novel_docs

        except Exception as e:
            logger.error(f"Error in novel document retrieval: {str(e)}")
            return []

    def _analyze_hop(
        self,
        original_query: str,
        previous_chunks: List[Document],
        new_chunks: List[Document],
    ) -> Dict:
        """Single LLM call for hop analysis."""
        try:
            previous_chunks_text = (
                "\n".join(
                    f"PREVIOUS CHUNK {i+1}:\n{doc.content}"
                    for i, doc in enumerate(previous_chunks)
                )
                if previous_chunks
                else "NO PREVIOUS CHUNKS"
            )

            new_chunks_text = "\n".join(
                f"NEW CHUNK {i+1}:\n{doc.content}" for i, doc in enumerate(new_chunks)
            )

            chain = self.analysis_prompt | self.llm | JsonOutputParser()

            response = chain.invoke(
                {
                    "original_query": original_query,
                    "previous_chunks": previous_chunks_text,
                    "new_chunks": new_chunks_text,
                }
            )

            return response

        except Exception as e:
            logger.error(f"Error in hop analysis: {str(e)}")
            return {
                "novelty_analysis": ["Error in analysis"],
                "deficiency_analysis": ["Error in analysis"],
                "refined_query": original_query,
            }

    def process_user_query(
        self,
        query: str,
        rerank: bool = True,
        num_hops: int = 3,
        chunks_per_hop: int = 3,
        llm: str = "gpt-4o",
        useMetadata: bool = True,
        domain: str = "finance",
        filter: str = "",
    ) -> Dict:
        """Process user query through multi-hop retrieval and return final context and analysis."""

        self.domain = domain
        self._setup_llm(llm)
        hops = []
        current_query = query
        doc_strings = set()
        doc_list = []

        logger.info(f"Metadata filter: {filter}")

        # Initialize reranker if needed
        reranker = CohereReranker() if rerank else None

        for hop in range(num_hops):
            logger.info(f"Executing hop {hop + 1}")

            try:
                # Get new documents for the hop
                new_docs = self._get_novel_documents(
                    current_query,
                    doc_strings,
                    k=chunks_per_hop + hop,
                    metadata_filter=filter,
                )

                # Analyze the hop
                analysis = self._analyze_hop(
                    original_query=query, previous_chunks=doc_list, new_chunks=new_docs
                )

                # Create hop record
                hop_record = RetrievalHop(
                    query=current_query,
                    retrieved_documents=new_docs,
                    analysis=analysis,
                    refined_query=analysis.get("refined_query"),
                )

                hops.append(hop_record)

                # Update tracking
                doc_list.extend(new_docs)
                doc_strings.update(doc.content for doc in new_docs)
                current_query = analysis.get("refined_query", current_query)

            except Exception as e:
                logger.error(f"Error in hop {hop + 1}: {str(e)}")
                break

        # Process final documents
        final_docs = list(doc_strings)  # Initialize with original documents
        if doc_strings and rerank:
            try:
                reranked_docs = reranker.rerank(
                    query, list(doc_strings), top_k=min(5, len(doc_strings))
                )
                if reranked_docs:  # Only update if reranking was successful
                    final_docs = reranked_docs
            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}")
                # Keep using final_docs (original documents) as fallback

        # Process context
        processed_contexts = []
        for doc in final_docs:
            processed_contexts.append(doc)

        context = (
            " ".join(processed_contexts)
            if processed_contexts
            else " ".join(doc_strings)
        )
        logger.info(f"Final context length: {len(context)}")
        logger.info(f"Final context avg tokens: {len(context)/4}")

        # Return results
        return {
            "context": context,
            "hops": [
                {
                    "query": hop.query,
                    "documents": [doc.content for doc in hop.retrieved_documents],
                    "analysis": hop.analysis,
                    "refined_query": hop.refined_query,
                }
                for hop in hops
            ],
            "metadata_filter": filter,
        }


if __name__ == "__main__":
    retriever = MultiHopRetriever()
    query = "Consider the Manufacturing and Supply Agreement between Vapotherm, Inc. and Medica, S.p.A.; Is there an anti-assignment clause in this contract"
    response = retriever.process_user_query(
        query,
        rerank=True,
        num_hops=3,
        chunks_per_hop=3,
        llm="gpt-4o-mini",
    )
    print(json.dumps(response, indent=2))
