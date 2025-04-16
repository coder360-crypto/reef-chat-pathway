import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from typing import Any, Dict, List, Tuple

from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from utils.metadata_generator import MetadataGenerator
from utils.moray_utils.joiner import Joiner
from utils.moray_utils.tool_descriptions import (
    PLANNER_TOOL_DESCRIPTIONS,
    REPLANNER_TOOL_DESCRIPTIONS,
)
from utils.tools.compliance_checker import ComplianceChecker
from utils.tools.esg_comparison import ESGAnalyst
from utils.tools.financial_analyst import (
    CorporateActionsAgent,
    FinancialContextAnalyst,
    FinancialMetricsScoresAgent,
    FinancialNewsAgent,
    FinancialStatementAnalysisAgent,
    MacroeconomicAnalysisAgent,
    MarketPriceTradingAgent,
    TechnicalIndicatorAnalyst,
    search_symbol,
)
from utils.tools.graph_tool import GraphTool
from utils.tools.ipr_agent import IPRAgent
from utils.tools.legal_analyst import LegalAnalyst
from utils.tools.math_tools import get_math_tool
from utils.tools.multihop_tool import MultiHopTool
from utils.tools.pdf_read_tool import PDFQueryTool
from utils.tools.search_tools import JinaSearchTool, SerperSearchTool, TavilySearchTool
from utils.tools.similar_case import SimilarCaseToolHelper
from utils.tools.valuation_tool import ValuationTool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ToolManager:
    """Manages the initialization and caching of tools."""

    _tool_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self.common_tools: Dict[str, BaseTool] = {}
        self.user_tools: Dict[str, BaseTool] = {}
        self.tools: List[BaseTool] = []
        self.use_internet = True
        self.llm = None
        self.BASE_PLANNER_TOOL_DESCRIPTIONS = PLANNER_TOOL_DESCRIPTIONS.copy()
        self.BASE_REPLANNER_TOOL_DESCRIPTIONS = REPLANNER_TOOL_DESCRIPTIONS.copy()

    def _set_custom_tool_descriptions(self, doc_list: List[str]):
        """
        Set custom tool descriptions based on relevant documents found for the query.

        Args:
            doc_list: List of relevant documents found for the query
        """
        # Format the doc_list for replanner tool description
        formatted_doc_list_json = (
            "\n".join(
                [
                    f"- {key}: Key Entities: {', '.join(value)}"
                    for key, value in doc_list.items()
                ]
            )
            if doc_list
            else "No relevant documents found related to the query."
        )
        self.BASE_PLANNER_TOOL_DESCRIPTIONS[
            "multihop_tool"
        ] = self.BASE_PLANNER_TOOL_DESCRIPTIONS["multihop_tool"].format(
            doc_list=(
                formatted_doc_list_json
                if doc_list
                else "No relevant documents found related to the query."
            )
        )

        self.BASE_REPLANNER_TOOL_DESCRIPTIONS[
            "multihop_tool"
        ] = self.BASE_REPLANNER_TOOL_DESCRIPTIONS["multihop_tool"].format(
            doc_list=(
                formatted_doc_list_json
                if doc_list
                else "No relevant documents found related to the query."
            )
        )

    def initialize_tools(
        self,
        llm: BaseLanguageModel,
        selected_docs: List[str],
        jwt_token: str,
        temp_files: List[str],
        use_internet: bool,
        query: str,
        context: str = None,
    ):
        """
        Initialize all tools based on configuration and cache status.

        Args:
            llm: Language model to use for tools
            selected_docs: List of documents selected for analysis
            jwt_token: Authentication token for API access
            temp_files: List of temporary files to process
            use_internet: Flag to enable/disable internet-dependent tools
        """
        # Store configuration
        self.use_internet = use_internet
        logger.info(f"Use internet: {self.use_internet}")
        self.llm = llm
        self.temp_files = temp_files
        self.selected_docs = selected_docs
        self.jwt_token = jwt_token
        self.query = query
        self.context = context
        self.filter, self.doc_list = MetadataGenerator(
            jwt_token=self.jwt_token,
            selected_docs=self.selected_docs,
            llm=self.llm,
        ).get_filter_and_relevant_docs(
            query=(
                self.query + "\nContext: " + " ".join(self.context)
                if self.context
                else self.query
            )
        )

        if self.doc_list:
            self.initialize_multihop_tool(llm)
            self._set_custom_tool_descriptions(self.doc_list)
        logger.info("Initializing common tools")
        self._initialize_common_tools(llm)

        # Initialize user-specific tools and combine all tools
        self._initialize_user_tools(llm, self.filter, temp_files)
        self.tools = list(self.common_tools.values())
        self.tools.extend(list(self.user_tools.values()))

        # Validate and build tool descriptions
        self._validate_tools()
        self._build_tool_descriptions()

    def _initialize_common_tools(self, llm: BaseLanguageModel):
        """
        Initialize common tools based on internet access configuration.

        Args:
            llm: Language model to use for tools

        Note:
            When use_internet is False, only basic non-internet tools are initialized.
            Internet-dependent tools are only initialized when use_internet is True.
        """
        # Initialize basic tools that don't require internet
        self.calculate = get_math_tool(llm)
        self.legal_analyst = LegalAnalyst(llm).get_legal_tool()
        self.graph_tool = GraphTool(llm)
        self.financial_context_tool = FinancialContextAnalyst(
            llm
        ).get_financial_context_tool()
        # Initialize the common_tools dictionary with non-internet tools
        self.common_tools = {
            "calculate": self.calculate,
            "legal_analyst": self.legal_analyst,
            "graph_tool": self.graph_tool,
            "financial_context_tool": self.financial_context_tool,
        }

        # Only initialize internet-dependent tools if enabled
        if self.use_internet:
            self._initialize_internet_tools(llm)

    def _initialize_internet_tools(self, llm: BaseLanguageModel):
        """
        Initialize tools that require internet access.

        Args:
            llm: Language model to use for tools
        """
        # Initialize search tools
        self.tavily_search_results_json = TavilySearchTool(
            max_results=5, search_depth="advanced"
        ).get_tool()
        self.serper_search_agent = SerperSearchTool().get_tool()
        self.jina_search_agent = JinaSearchTool().get_tool()
        self.wikipedia_agent = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.search_symbol = search_symbol

        # Initialize financial analysis tools
        self.trading_analyst = MarketPriceTradingAgent(llm).get_trading_tool()
        self.macroeconomics_tool = MacroeconomicAnalysisAgent(
            llm
        ).get_macroeconomic_tool()
        self.corporateactions_tool = CorporateActionsAgent(
            llm
        ).get_corporate_actions_tool()
        self.financialstatement_tool = FinancialStatementAnalysisAgent(
            llm
        ).get_financial_statement_tool()
        self.financialnews_tool = FinancialNewsAgent(llm).get_news_tool()
        self.technicalindicators_tool = TechnicalIndicatorAnalyst(
            llm
        ).get_technical_tool()
        self.financial_metrics_tool = FinancialMetricsScoresAgent(
            llm
        ).get_financial_metrics_tool()
        self.valuation_tool = ValuationTool(llm).get_valuation_tool()
        self.ipr_analyst = IPRAgent(llm, self.temp_files).get_ipr_analysis_tool()
        # Add internet-dependent tools to common_tools
        internet_tools = {
            "tavily_search_results_json": self.tavily_search_results_json,
            "wikipedia_agent": self.wikipedia_agent,
            "search_symbol": self.search_symbol,
            "trading_analyst": self.trading_analyst,
            "macroeconomics_tool": self.macroeconomics_tool,
            "corporateactions_tool": self.corporateactions_tool,
            "financialstatement_tool": self.financialstatement_tool,
            "financialnews_tool": self.financialnews_tool,
            "technicalindicators_tool": self.technicalindicators_tool,
            "valuation_tool": self.valuation_tool,
            "serper_search_agent": self.serper_search_agent,
            "jina_search_agent": self.jina_search_agent,
            "financial_metrics_tool": self.financial_metrics_tool,
            "ipr_analyst": self.ipr_analyst,
        }
        self.common_tools.update(internet_tools)

    def _initialize_user_tools(
        self,
        llm: BaseLanguageModel,
        filter: str,
        temp_files: List[str],
    ):
        if temp_files:
            self.compliance_checker = ComplianceChecker(
                llm, temp_files
            ).get_compliance_tool()
            self.pdf_query_tool = PDFQueryTool()
            self.user_tools["compliance_checker"] = self.compliance_checker
            self.user_tools["pdf_query_tool"] = self.pdf_query_tool
            self.BASE_PLANNER_TOOL_DESCRIPTIONS["pdf_query_tool"] = (
                self.BASE_PLANNER_TOOL_DESCRIPTIONS["pdf_query_tool"].format(
                    pdf_list="\n".join(self.temp_files or [])
                )
            )
            if self.use_internet:
                self.esg_analyst = ESGAnalyst(
                    llm, self.temp_files
                ).get_esg_analysis_tool()
                self.similar_case_finder = SimilarCaseToolHelper(
                    llm
                ).get_similar_case_tool(temp_files=self.temp_files)
                self.user_tools["similar_case_finder"] = self.similar_case_finder
                self.user_tools["esg_analyst"] = self.esg_analyst

    def initialize_multihop_tool(self, llm: BaseLanguageModel):
        self.multihop_tool = MultiHopTool(llm, query=self.query, filter=self.filter)
        self.user_tools["multihop_tool"] = self.multihop_tool

    def _validate_tools(self):
        """
        Validate tools and their descriptions for both planner and replanner.

        Args:
            tools: Sequence of tools to validate
            planner_descriptions: Dictionary of tool descriptions for initial planning
            replanner_descriptions: Dictionary of tool descriptions for replanning

        Raises:
            ValueError: If tools are empty or invalid
        """
        # Check if tools exist
        if not self.tools:
            raise ValueError("Tools cannot be empty")

        # Validate description dictionaries
        if not self.BASE_PLANNER_TOOL_DESCRIPTIONS:
            raise ValueError("Planner descriptions dictionary cannot be empty")
        if not self.BASE_REPLANNER_TOOL_DESCRIPTIONS:
            raise ValueError("Replanner descriptions dictionary cannot be empty")

        # Validate tool descriptions format
        for tool in self.tools:
            if (
                tool.name not in self.BASE_PLANNER_TOOL_DESCRIPTIONS
                and tool.name not in self.BASE_REPLANNER_TOOL_DESCRIPTIONS
            ):
                raise ValueError(
                    f"Tool {tool.name} is not in the TOOL_DESCRIPTIONS dictionary"
                )

    def _build_tool_descriptions(self) -> Tuple[List[str], List[str]]:
        self.planner_descriptions = []
        self.replanner_descriptions = []
        for tool in self.tools:
            if self.BASE_PLANNER_TOOL_DESCRIPTIONS.get(tool.name):
                self.planner_descriptions.append(
                    f"{tool.name} : {self.BASE_PLANNER_TOOL_DESCRIPTIONS.get(tool.name)}/n"
                )
            if self.BASE_REPLANNER_TOOL_DESCRIPTIONS.get(tool.name):
                self.replanner_descriptions.append(
                    f"{tool.name} : {self.BASE_REPLANNER_TOOL_DESCRIPTIONS.get(tool.name)}/n"
                )

        print(f"Planner descriptions: {self.planner_descriptions}")

        self.planner_tool_names = [
            tool.name
            for tool in self.tools
            if tool.name in self.BASE_PLANNER_TOOL_DESCRIPTIONS
        ]
        logger.info(f"Planner tool names: {self.planner_tool_names}")
        self.replanner_tool_names = [
            tool.name
            for tool in self.tools
            if tool.name in self.BASE_REPLANNER_TOOL_DESCRIPTIONS
        ]
        self.planner_tools = [
            tool
            for tool in self.tools
            if tool.name in self.BASE_PLANNER_TOOL_DESCRIPTIONS
        ]
        self.replanner_tools = [
            tool
            for tool in self.tools
            if tool.name in self.BASE_REPLANNER_TOOL_DESCRIPTIONS
        ]


# def _load_cached_tools(self, cached_tools):
#     """Load tools from cache"""
#     logger.info("Loading cached tools")
#     for tool_name, tool in cached_tools.items():
#         setattr(self, tool_name, tool)

# @classmethod
# def clear_cache(cls):
#     """Clear the tool cache if needed"""
#     cls._tool_cache.clear()
