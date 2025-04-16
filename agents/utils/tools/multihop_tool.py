import logging
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Any, Dict, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from services.multihop_service import MultiHopRetriever

logger = logging.getLogger(__name__)


class RetrieverSchema(BaseModel):
    """Schema for Retriever Tool input.

    Args:
        query (str): The query string to retrieve information for
        context (Optional[str]): Additional context to help with retrieval
        num_hops (Optional[int]): Number of retrieval hops to perform, defaults to 3
    """

    query: str = Field(
        ...,
        description="The query to retrieve information for",
    )
    context: Optional[str] = Field(
        None, description="Additional context to help with retrieval"
    )
    num_hops: Optional[int] = Field(
        3, description="Number of retrieval hops to perform"
    )


class MultiHopTool(BaseTool):
    """Tool for multi-hop information retrieval.

    A tool that performs multi-hop retrieval to gather comprehensive information
    and find connections between different pieces of data.

    Args:
        llm (Any): Language model instance
        query (str): Initial query string
        filter (str): Filter criteria for retrieval
    """

    name: str = "multihop_tool"
    description: str = """
        Retrieve and analyze information through multi-hop retrieval. This tool gives you access to data not found on the internet.
        Use this tool when you need to:
        1. Gather comprehensive information about a topic
        2. Find connections between different pieces of information
        3. Get detailed context for complex queries
        4. Analyze information gaps and gather missing details
        """
    description: str = """
        Retrieve and analyze information through multi-hop retrieval. This tool gives you access to data not found on the internet.
        Use this tool when you need to:
        1. Gather comprehensive information about a topic
        2. Find connections between different pieces of information
        3. Get detailed context for complex queries
        4. Analyze information gaps and gather missing details
        """
    args_schema: type[BaseModel] = RetrieverSchema
    llm: Any = Field(default=None)
    retriever: MultiHopRetriever = Field(default_factory=lambda: None)
    filter: str = ""
    query: str = ""

    def __init__(
        self,
        llm: Any,
        query: str = "",
        filter: str = "",
    ) -> None:
        """Initialize the retriever tool."""
        super().__init__(llm=llm)
        self.llm = llm

        self.retriever = MultiHopRetriever()
        self.filter = filter
        self.query = query

    def _run(
        self,
        query: str,
        context: Optional[str] = None,
        num_hops: int = 3,
        **kwargs,  # Add kwargs to handle additional parameters
    ) -> str:
        """Run the retriever tool.

        Args:
            query (str): Query string to process
            context (Optional[str]): Additional context
            num_hops (int): Number of retrieval hops
            **kwargs: Additional parameters

        Returns:
            str: Formatted string containing thought process and retrieved context
        """
        try:
            self.query = query
            results = self.retriever.process_user_query(
                query=self.query,
                rerank=True,
                # num_hops=num_hops,
                num_hops=2,
                chunks_per_hop=7,
                filter=self.filter,
            )

            thought_process = self._format_thought_process(results)
            return f"""Thought Process: {thought_process}\n\nRetrieved Context: {results['context']}"""
        except Exception as e:
            logger.error(f"Error in retriever tool: {str(e)}")
            return f"Error retrieving information: {str(e)}"

    async def _arun(
        self,
        query: str,
        context: Optional[str] = None,
        num_hops: int = 3,
        **kwargs,  # Add kwargs to handle additional parameters including domain
    ) -> str:
        """Async implementation of the tool."""
        return self._run(query, context, num_hops, **kwargs)

    def _format_thought_process(self, hop_info: Dict) -> str:
        """Format the thought process from retrieval hops.

        Args:
            hop_info (Dict): Dictionary containing hop information

        Returns:
            str: Formatted string of the thought process
        """
        final_hop = hop_info["hops"][-1]
        thoughts = []
        thoughts.append("New Information Found:")
        for insight in final_hop["analysis"]["novelty_analysis"]:
            thoughts.append(f"- {insight}")
        thoughts.append(f"\nRefined Query: {final_hop['refined_query']}")
        thoughts.append("-" * 50)

        return "\n".join(thoughts)


if __name__ == "__main__":
    tool = MultiHopTool(llm=None)
    tool.run("What is the Best Buy revenue for 2022?")
