# Standard library imports
import os
import platform
import sys
from datetime import datetime

# Third-party imports
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List

# Local imports and configurations
from config import AgentsConfig as Config
from langchain.callbacks import get_openai_callback
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from utils.code_architect_utils.final_formatter import final_formatter
from utils.code_architect_utils.logic_tool import logic_tool
from utils.code_architect_utils.prompts import MAIN_PROMPT
from utils.code_architect_utils.tool_docstrings import TOOL_DOCSTRINGS
from utils.metadata_generator import MetadataGenerator
from utils.tools.financial_analyst import (
    MacroeconomicAnalysisAgent,
    MarketPriceTradingAgent,
    TechnicalIndicatorAnalyst,
    search_symbol,
)
from utils.tools.legal_analyst import LegalAnalyst
from utils.tools.math_tools import get_math_tool
from utils.tools.multihop_tool import MultiHopTool
from utils.tools.search_tools import SerperSearchTool


class CARP:
    """
    Code Architect for Planning (CArP) for handling complex queries using multiple tools and agents.
    
    Args:
        llm (str): Language model identifier, defaults to "gpt-4o"
        jwt_token (str): Authentication token for API access
        selected_docs (List[str]): List of selected documents to process
    """
    def __init__(
        self, llm: str = "gpt-4o", jwt_token: str = None, selected_docs: List[str] = []
    ):

        if not jwt_token:
            raise ValueError("JWT token is required")

        # TODO add more tools
        self.llm = ChatOpenAI(
            model=llm, api_key=Config().OPENAI_API_KEY, temperature=0.1
        )

        self.metadata_generator = MetadataGenerator(
            jwt_token=jwt_token, selected_docs=selected_docs, llm=self.llm
        )

        self.legal_analyst_class = LegalAnalyst(llm=self.llm)
        self.math_tool_class = get_math_tool(llm=self.llm)
        self.technical_indicator_analyst_class = TechnicalIndicatorAnalyst(llm=self.llm)
        self.macroeconomic_analyst_class = MacroeconomicAnalysisAgent(llm=self.llm)
        self.financial_news_analyst_class = YahooFinanceNewsTool()
        self.trading_analyst_class = MarketPriceTradingAgent(llm=self.llm)
        self.search_symbol = search_symbol
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.serper_search_class = SerperSearchTool().get_tool()

    def _compile_into_plan(
        self, query: str, failed_plan: str, reason_for_failure: str
    ) -> str:
        """
        Compiles user query into an executable plan using LLM.
        
        Args:
            query (str): User's input query
            failed_plan (str): Previous failed plan if any
            reason_for_failure (str): Reason for previous plan failure
            
        Returns:
            str: Compiled execution plan
        """

        failed_plan_content = (
            f"""
                <last_failed_plan>{failed_plan}</last_failed_plan>
                <reason_for_failure>{reason_for_failure}</reason_for_failure>
            """
            if failed_plan
            else ""
        )

        response = self.llm.invoke(
            input=[
                {
                    "role": "system",
                    "content": MAIN_PROMPT,
                },
                {
                    "role": "user",
                    "content": failed_plan_content
                    + f"""
                        <query>{query}</query>
                        <tools>
                        {" ".join(TOOL_DOCSTRINGS[self.use_retriever_tool:])}
                        </tools>
                    """,
                },
            ],
        )

        return response.content

    def process_user_query(
        self,
        query: str,
        context: str = None,
        temp_files: list = None,
    ) -> Dict[str, Any]:
        """
        Processes user query and returns results with execution metrics.
        
        Args:
            query (str): User's input query
            context (str, optional): Additional context for the query
            temp_files (list, optional): List of temporary files to process
            
        Returns:
            Dict[str, Any]: Results including final response, callbacks, metrics
        """
        execution_start = datetime.now()
        callbacks = []
        results = []

        if context:
            query = query + "\n Context: " + context

        if temp_files:
            query = (
                query
                + "Here are the attached files that can be used: "
                + str(temp_files)
            )

        (
            self.filter,
            self.doc_list,
        ) = self.metadata_generator.get_filter_and_relevant_docs(query)

        self.multihop_tool_class = MultiHopTool(
            llm=self.llm, filter=self.filter, query=query
        )

        # import pdb

        # pdb.set_trace()

        # print(self.doc_list)

        self.use_retriever_tool = int(len(self.doc_list) == 0)

        # graph_tool = self.graph_tool_class._run
        legal_analyst = self.legal_analyst_class.get_legal_tool().func
        solve_tool = self.math_tool_class.func
        retriever_tool = self.multihop_tool_class._run
        technical_indicator_analyst = (
            self.technical_indicator_analyst_class.get_technical_tool().func
        )
        macroeconomic_analyst = (
            self.macroeconomic_analyst_class.get_macroeconomic_tool().func
        )
        financial_news_analyst = self.financial_news_analyst_class._run
        trading_analyst = self.trading_analyst_class.get_trading_tool().func
        search_symbol = self.search_symbol
        wikipedia = self.wikipedia_tool._run
        serper_search_class = self.serper_search_class._run

        failed_plan = None
        reason_for_failure = None
        with get_openai_callback() as cb:
            for attempt in range(2):
                try:
                    step_start = datetime.now()
                    plan = self._compile_into_plan(
                        query, failed_plan, reason_for_failure
                    )

                    # Record planning step
                    callbacks.append(
                        {
                            "step": "planning",
                            "timestamp": step_start.isoformat(),
                            "total_tokens": cb.total_tokens,
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_cost": cb.total_cost,
                            "execution_time": (
                                datetime.now() - step_start
                            ).total_seconds(),
                        }
                    )

                    plan = plan[
                        plan.find("<response>")
                        + len("<response>") : plan.find("</response>")
                    ]

                    plan = plan.strip("```python")
                    plan = plan.strip("`")

                    print(plan)

                    # Create a namespace dictionary to share variables
                    local_namespace = {
                        # "graph_tool": graph_tool,
                        "legal_analyst": legal_analyst,
                        "solve_tool": solve_tool,
                        "retriever_tool": retriever_tool,
                        "technical_indicator_analyst": technical_indicator_analyst,
                        "macroeconomic_analyst": macroeconomic_analyst,
                        "financial_news_analyst": financial_news_analyst,
                        "trading_analyst": trading_analyst,
                        "search_symbol": search_symbol,
                        "wikipedia": wikipedia,
                        "financial_news_analyst": financial_news_analyst,
                        "serper_search_class": serper_search_class,
                        "logic_tool": logic_tool,
                    }

                    # Execute the plan in the shared namespace
                    exec(plan, globals(), local_namespace)

                    # Get output from the namespace
                    output = local_namespace.get("output")
                    print(output)

                    # import pdb; pdb.set_trace()

                    if not locals().get("output", None):
                        raise Exception("No output found")

                    step_start = datetime.now()
                    response = final_formatter(output, query)

                    # Record execution step
                    callbacks.append(
                        {
                            "step": "execution",
                            "timestamp": step_start.isoformat(),
                            "total_tokens": cb.total_tokens,
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                            "total_cost": cb.total_cost,
                            "execution_time": (
                                datetime.now() - step_start
                            ).total_seconds(),
                        }
                    )

                    results.append({"output": output, "response": response})
                    break

                except Exception as e:
                    reason_for_failure = str(e)
                    failed_plan = plan
                    print(e, "retrying...")

            execution_time = datetime.now() - execution_start

            return {
                "final_results": results[0]["response"],
                "callbacks": callbacks,
                "task_flow_data": {"plan": plan},
                "execution_metrics": {
                    "total_time": str(execution_time),
                    "total_steps": len(results),
                    "input_tokens": callbacks[-1]["prompt_tokens"] if callbacks else 0,
                    "output_tokens": (
                        callbacks[-1]["completion_tokens"] if callbacks else 0
                    ),
                    "total_tokens": callbacks[-1]["total_tokens"] if callbacks else 0,
                    "total_cost": callbacks[-1]["total_cost"] if callbacks else 0,
                    "system_info": {
                        "platform": platform.system(),
                        "python_version": platform.python_version(),
                        "memory_usage": psutil.Process().memory_info().rss
                        / 1024
                        / 1024,  # MB
                    },
                },
            }


if __name__ == "__main__":
    carp = CARP(llm="gpt-4o", jwt_token=Config().JWT_TOKEN)

    average_cost = 0
    print(
        carp.process_user_query(
            "Is 3M a capital-intensive business based on FY2022 data?"
        )
    )
