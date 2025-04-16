"""
Search tool implementations for different search providers.

This module provides structured tools for:
- Tavily Search API
- Serper (Google) Search API
- Jina Search API

Each tool handles:
- Query execution
- Response formatting
- Error handling
- Markdown output generation
"""
import os
import sys
import logging
from typing import Optional

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tavily import TavilyClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import AgentsConfig as Config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
config = Config()


class TavilySearchSchema(BaseModel):
    """Input schema for Tavily search queries."""
    
    query: str = Field(
        ..., 
        description="The search query to be executed."
    )
    context: Optional[str] = Field(
        None, 
        description="Additional context for the search query."
    )


class SerperSearchSchema(BaseModel):
    """Input schema for Serper search queries."""
    
    query: str = Field(
        ..., 
        description="The search query to be executed."
    )


class JinaSearchSchema(BaseModel):
    """Input schema for Jina search queries."""
    
    query: str = Field(
        ..., 
        description="The search query to be executed."
    )


class TavilySearchTool:
    """
    Implements search functionality using the Tavily API.
    
    Features:
    - Advanced search capabilities
    - Configurable result count
    - Structured markdown output
    - Detailed metadata
    """

    def __init__(self, max_results: int, search_depth: str):
        """
        Initialize Tavily search tool.
        
        Args:
            max_results: Maximum number of results to return
            search_depth: Depth of search ('basic' or 'advanced')
        """
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)
        self.max_results = max_results
        self.search_depth = search_depth

    def search(self, query: str):
        """
        Execute search query and format results.
        
        Args:
            query: Search query string
            
        Returns:
            Dict containing formatted message and metadata
        """
        try:
            # Execute search
            response = self.client.search(
                query, 
                max_results=self.max_results, 
                search_depth=self.search_depth
            )
            results = response.get('results', [])
            
            # Format results in markdown
            message_parts = [f"### Search Results for: {query}\n\n"]
            
            for result in results:
                title = result.get('title', 'No Title')
                url = result.get('url', '')
                content = result.get('content', 'No content available')
                
                message_parts.extend([
                    f"#### {title}\n",
                    f"{content}\n",
                    f"Source: [{url}]({url})\n\n"
                ])
            
            return {
                'message': "".join(message_parts),
                'metadata': {
                    'query': response.get('query', ''),
                    'response_time': response.get('response_time', 0),
                    'result_count': len(results),
                    'source': 'tavily_search'
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute Tavily search: {str(e)}")
            return JinaSearchTool().search(query)

    def get_tool(self) -> StructuredTool:
        """Get the configured search tool."""
        return StructuredTool(
            name="tavily_search_results_json",
            description="Retrieve current, general, or trending information from web sources using Tavily Search API.",
            func=self.search,
            args_schema=TavilySearchSchema
        )


class SerperSearchTool:
    """
    Implements search functionality using the Serper (Google) API.
    
    Features:
    - Google search results
    - Answer box support
    - Related questions
    - Structured markdown output
    """

    def search(self, query: str):
        """
        Execute search query and format results.
        
        Args:
            query: Search query string
            
        Returns:
            Dict containing formatted message and metadata
        """
        try:
            url = f"https://google.serper.dev/search?q={query}&apiKey={config.SERPER_API_KEY}"
            payload = {}
            headers = {}
            raw_response = requests.request("GET", url, headers=headers, data=payload).json()
            
            # Initialize message parts
            message_parts = []
            
            # Format answer box if present
            if 'answerBox' in raw_response:
                self._format_answer_box(raw_response['answerBox'], message_parts)
            
            # Format organic results
            if 'organic' in raw_response:
                self._format_organic_results(raw_response['organic'], message_parts)
            
            # Format related questions
            if 'peopleAlsoAsk' in raw_response:
                self._format_related_questions(raw_response['peopleAlsoAsk'], message_parts)
            
            # Format related searches
            if 'relatedSearches' in raw_response:
                self._format_related_searches(raw_response['relatedSearches'], message_parts)
            
            return {
                'message': "".join(message_parts),
                'metadata': {
                    'searchParameters': raw_response.get('searchParameters', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to execute Serper search: {str(e)}")
            return JinaSearchTool().search(query)

    def _format_answer_box(self, answer_box: dict, message_parts: list):
        """Format answer box section."""
        message_parts.extend([
            "### Featured Answer\n",
            f"**{answer_box.get('title', '')}**\n" if 'title' in answer_box else "",
            f"{answer_box.get('snippet', '')}\n" if 'snippet' in answer_box else "",
            f"Source: [{answer_box['link']}]({answer_box['link']})\n" if 'link' in answer_box else "",
            "\n"
        ])

    def _format_organic_results(self, results: list, message_parts: list):
        """Format organic search results section."""
        message_parts.append("### Search Results\n")
        for result in results:
            message_parts.extend([
                f"#### {result.get('title', 'No Title')}\n",
                f"{result.get('snippet', 'No snippet available')}\n",
                f"Source: [{result['link']}]({result['link']})\n" if 'link' in result else "",
                "\n"
            ])

    def _format_related_questions(self, questions: list, message_parts: list):
        """Format related questions section."""
        message_parts.append("### People Also Ask\n")
        for question in questions:
            message_parts.extend([
                f"- **Q: {question.get('question', '')}**\n",
                f"  A: {question.get('snippet', 'No answer available')}\n",
                f"  Source: [{question['link']}]({question['link']})\n" if 'link' in question else "",
            ])
        message_parts.append("\n")

    def _format_related_searches(self, searches: list, message_parts: list):
        """Format related searches section."""
        message_parts.append("### Related Searches\n")
        for search in searches:
            if query := search.get('query'):
                message_parts.append(f"- {query}\n")
        message_parts.append("\n")

    def get_tool(self) -> StructuredTool:
        """Get the configured search tool."""
        return StructuredTool(
            name="serper_search",
            description="Retrieve current, general, or trending information from web sources using Google Serper API.",
            func=self.search,
            args_schema=SerperSearchSchema,
        )


class JinaSearchTool:
    """
    Implements search functionality using the Jina API.
    
    Features:
    - Simple search interface
    - Basic error handling
    """

    def search(self, query: str):
        """Execute search query."""
        try:
            url = f"https://s.jina.ai/https://{query}"
            headers = {"Authorization": f"Bearer {config.JINA_API_KEY}"}
            response = requests.get(url, headers=headers)
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to execute Jina search: {str(e)}")
            return {"error": str(e)}

    def get_tool(self) -> StructuredTool:
        """Get the configured search tool."""
        return StructuredTool(
            name="jina_search",
            description="Retrieve current, general, or trending information from web sources using Jina Search API.",
            func=self.search,
            args_schema=JinaSearchSchema,
        )