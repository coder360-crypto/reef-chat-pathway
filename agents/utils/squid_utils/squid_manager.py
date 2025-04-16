import json
import logging
import os
import sys
import traceback
from typing import Annotated, Any, Dict, List, TypedDict

from config import AgentsConfig as Config

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from langchain.base_language import BaseLanguageModel
from langchain_community.tools import (
    BalanceSheets,
    CashFlowStatements,
    IncomeStatements,
    PolygonAggregates,
    PolygonFinancials,
    PolygonLastQuote,
    PolygonTickerNews,
    WikipediaQueryRun,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.financial_datasets import FinancialDatasetsAPIWrapper
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from utils.metadata_generator import MetadataGenerator
from utils.squid_utils.flow_tracker import TaskFlow
from utils.squid_utils.squid_planner import ThoughtBasedPlanner
from utils.squid_utils.squid_prompts import (
    AGENT_BASIC_CAPABILITIES,
    AGENT_CAPABILITIES,
    PromptStore,
)
from utils.squid_utils.squid_registry import (
    dcf_tool,
    get_actions,
    get_financials,
    get_historical_data,
    get_holders,
    get_news,
    get_option_chain,
    get_options,
    get_recommendations,
    get_shares_count,
    get_stock_info,
)
from utils.tools.compliance_checker import ComplianceChecker
from utils.tools.financial_analyst import (
    CorporateActionsAgent,
    FinancialNewsAgent,
    FinancialStatementAnalysisAgent,
    MacroeconomicAnalysisAgent,
    TechnicalIndicatorAnalyst,
)
from utils.tools.graph_tool import GraphTool
from utils.tools.ipr_agent import IPRAgent
from utils.tools.legal_analyst import LegalAnalyst
from utils.tools.multihop_tool import MultiHopTool
from utils.tools.pdf_read_tool import PDFQueryTool
from utils.tools.search_tools import JinaSearchTool, SerperSearchTool

config = Config()


# Initialize Polygon.io API tools for financial data
polygon = PolygonAPIWrapper()
poly_tools = [
    PolygonLastQuote(api_wrapper=polygon),
    PolygonTickerNews(api_wrapper=polygon),
    PolygonFinancials(api_wrapper=polygon),
    PolygonAggregates(api_wrapper=polygon),
]

# Initialize Financial Datasets API tools
api_wrapper = FinancialDatasetsAPIWrapper()
findata_tools = [
    IncomeStatements(api_wrapper=api_wrapper),
    BalanceSheets(api_wrapper=api_wrapper),
    CashFlowStatements(api_wrapper=api_wrapper),
]

# Initialize Yahoo Finance tools for market data
yf_tools = [
    get_actions,
    get_financials,
    get_historical_data,
    get_holders,
    get_news,
    get_option_chain,
    get_options,
    get_recommendations,
    get_shares_count,
    get_stock_info,
]


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Custom function to merge two dictionaries"""
    merged = dict1.copy()
    merged.update(dict2)
    return merged


# State class to track conversation and execution state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_tasks: List[str]
    specialist_responses: Annotated[dict, merge_dicts]
    selected_agents: List[str]
    iteration_modes: Dict[str, int]
    iteration_state: Annotated[Dict[str, int], merge_dicts]
    execution_state: Annotated[Dict[str, str], merge_dicts]
    query_complexity: str
    response_length: str


# Model for specialist responses including analysis and plotting data
class SpecialistResponse(BaseModel):
    """Model for specialist's response including deficiency analysis"""

    Final_Response: str = Field(
        description="The final response from the specialist analyzing tool outputs"
    )
    Plottable_Information: str = Field(
        description="Information that can be plotted, with data details, units, and scales"
    )
    Deficiency_Analysis: str = Field(
        description="Analysis of any deficiencies in addressing key tasks and entities"
    )


# Individual specialist agent that can use tools and analyze data
class SQUIDSpecialist:
    """
    A specialist agent capable of using specific tools to analyze data and provide expert insights.
    
    Attributes:
        name (str): Name of the specialist
        manager (SQUIDManager): Reference to parent manager
        prompt_reg (PromptStore): Prompt registry for accessing templates
        system_prompt (str): System prompt defining specialist behavior
        llm (BaseLanguageModel): Language model with bound tools
        tool_node (ToolNode): Node for executing tools
        tools (List[BaseTool]): List of available tools
        response_compiler (BaseLanguageModel): LLM configured for structured outputs
    """

    def __init__(
        self, name: str, llm: BaseLanguageModel, tools: List[BaseTool], manager
    ):
        self.name = name
        self.manager = manager
        self.prompt_reg = self.manager.prompt_reg
        self.system_prompt = self._get_system_prompt()
        self.llm = llm.bind_tools(tools)
        self.tool_node = ToolNode(tools)
        self.tools = tools

        # Configure LLM for structured outputs
        self.response_compiler = self.llm.with_structured_output(SpecialistResponse)

    def _get_system_prompt(self) -> str:
        """System prompts for specialists"""
        warnings = self.prompt_reg._return_specialist_common_warnings()
        if not warnings:
            print("%" * 50)
            print("[WARNING] Specialist common warnings not found")

        prompts = self.prompt_reg._return_specialist_sysprompts(warnings=warnings)
        if not prompts:
            print("%" * 50)
            print("[WARNING] Specialist system prompts not found")

        return prompts.get(self.name, "You are a swarm specialist.")

    """
    Args:
        tool_response (AIMessage): Message containing tool calls to execute
    Returns:
        List[dict]: List of tool execution results with tool name, content and status
    """
    async def _execute_tools(self, tool_response: AIMessage) -> List[dict]:
        """Execute tools and collect results"""
        print(f"\n[{self.name}] Executing tools...")

        if hasattr(tool_response, "tool_calls") and tool_response.tool_calls:
            print(f"\n[{self.name}] Executing {len(tool_response.tool_calls)} tools:")
            for tool_call in tool_response.tool_calls:
                print(f"-> Tool: {tool_call.get('name')}")

            collected_info = []
            tool_results = await self.tool_node.ainvoke({"messages": [tool_response]})

            for msg in tool_results["messages"]:
                status = (
                    "error"
                    if (hasattr(msg, "status") and msg.status == "error")
                    else "success"
                )
                print(
                    f"[{self.name}] {'✓' if status == 'success' else '❌'} {msg.name}"
                )

                collected_info.append(
                    {"tool": msg.name, "content": msg.content, "status": status}
                )

                # Record in TaskFlow
                try:
                    tool_call = next(
                        call
                        for call in tool_response.tool_calls
                        if call.get("id") == msg.tool_call_id
                    )
                    self.manager.current_task_flow.add_tool_usage(
                        specialist_name=self.name,
                        tool_name=msg.name,
                        input_args=tool_call.get("args", {}),
                        output=msg.content,
                    )
                except Exception as e:
                    print(f"[{self.name}] Warning: Could not record tool usage: {e}")

            return collected_info
        return []

    """
    Args:
        state (State): Current conversation and execution state
    Returns:
        dict: Updated specialist responses with analysis results
    """
    async def analyze(self, state: State) -> dict:
        # Main analysis loop:
        # 1. Get task details
        # 2. Select and execute tools
        # 3. Analyze results
        # 4. Handle deficiencies if found
        """Process specialist task with deficiency handling"""
        print(f"\n{'#'*50}")
        print(f"[{self.name}] Starting analysis...")

        try:
            # Get task details
            tasks_dict = state["current_tasks"][-1]
            task_data = tasks_dict[self.name]

            full_task = (
                f"Task Description: {task_data['task_description']}\n"
                f"Key Entities to Analyze: {', '.join(task_data['key_entities'])}\n"
                f"Specific Tasks: {', '.join(task_data['key_tasks'])}"
            )

            print(f"[{self.name}] Task: {task_data['task_description'][:150]}...")
            print(f"[{self.name}] Entities: {', '.join(task_data['key_entities'])}")
            print(f"[{self.name}] Tasks: {', '.join(task_data['key_tasks'])}")

            # Initial tool selection using existing human_analysis_prompt
            print(f"\n[{self.name}] Selecting tools...")
            base_prompt = self.prompt_reg._return_human_analysis_prompt(task=full_task)
            tool_selection_prompt = (
                base_prompt
                + """ORDERS FROM USER: 
                - If your task explicitly mentions use of multihop tool AND tells you to use it, the DONT DARE use any other tools. 
                - This is because if the response made by using multihop deems to not be useful(rarely the case if task specifies multihop usage), then you will get a choice to rechoose so dont worry and believe tools enlisted by the task delegator.
                - Note that AT MAX, your current tool choice should have 1 extra tool of choice on top of the tools suggested by user.
                - Note that if user doesnt suggest any thing specifically like multihop, then you are free to do optimal tool choices based on guideleines given to you. Do efifcient and well thought out tool usage. If it deems to be less, then you will get another chance to work on tool choice making got it?
                """
            )
            tool_response = await self.llm.ainvoke(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=tool_selection_prompt),
                ]
            )

            # Execute initial tools
            tool_results = await self._execute_tools(tool_response)

            # Initial compilation with deficiency check
            print(f"\n[{self.name}] Compiling initial results...")
            base_compile_prompt = self.prompt_reg._return_spec_analysis_prompt(
                name=self.name, task=full_task, collected_info=tool_results
            )
            compile_prompt = (
                base_compile_prompt
                + """
Additional Analysis Requirements:
1. Deficiency Analysis: 
   - ENSURE TO ADD A FEW STATEMENTS OF DEFICIENCY ANALYSIS IN THE APPROPRIATEOUTPUT FIELD OF YOUR RESPONSE. the points below explain what it should be made of
   - Check if all key entities were adequately analyzed 
   - Verify if all specific tasks were completed
   - Format flag as <<DEFICIENCY TRUE>> if deficiency is detected based on above explained factors or <<DEFICIENCY FALSE>> if not found
   - If deficient, explain which aspects need more data/analysis
   - Also ensure to NAME AND MENTION FAILING/ ILL WORKING TOOLS, if any
2. Maintain Response Quality:
   - Follow all existing guidelines for accuracy and completeness
   - Ensure thorough coverage of all task components
"""
            )
            initial_response = None
            try:
                initial_response = await self.response_compiler.ainvoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=compile_prompt),
                    ]
                )
            except ValidationError as e:
                print(f"Encountered validation of fields error: {e}")
                print("[Retrying again.....]")
                initial_response = await self.response_compiler.ainvoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=compile_prompt),
                    ]
                )

            initial_response_str = f"""Response: {initial_response.Final_Response}
                                       Plottable Information: {initial_response.Plottable_Information}"""

            # Handle deficiency if detected
            if "<<DEFICIENCY TRUE>>" in initial_response.Deficiency_Analysis:
                print(f"\n[{self.name}] Deficiency detected. Selecting new tools...")

                # Tool reselection using modified human_analysis_prompt
                reselection_prompt = (
                    base_prompt
                    + f"""
Previous Tool Results and Deficiencies:
{initial_response.Deficiency_Analysis}

Focus for Tool Reselection:
- Select tools specifically to address identified gaps
- Dont use failed/ not useful tools that were mentioned in the above given deficiency analysis
- Ensure new tools complement previous information
- Ensure choosing atleast the same number or more number of tools than enlisted in deficiency analysis
- Tavily fails a lot, so switch to jina search tool and serper tool instead
- IN CASE OF complex/long work just rely on multihop retriever and search tools like tavily/jina/serper/wikipedia alone. Use other sources of information as well
- A variety of well chosen tools result in good results
- IF any tools, including "multihop" proved not useful, according to deficiency analysis, then you can decide to not use those tools this time
- Consider tools for verification of critical points
"""
                )
                new_tool_response = await self.llm.ainvoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=reselection_prompt),
                    ]
                )

                # Execute new tools
                new_results = await self._execute_tools(new_tool_response)

                # Final compilation with all results
                print(f"\n[{self.name}] Generating final compilation...")
                final_compile_prompt = (
                    self.prompt_reg._return_spec_analysis_prompt(
                        name=self.name, task=full_task, collected_info=new_results
                    )
                    + f"""
Previous Response that you made for the task is:
{initial_response_str}
                
Additional Requirements for Final Analysis:
1. Integration:
   - Combine insights from tool executions found
   - Combine insights from initial response you made an iteration ago
   - Ensure comprehensive coverage of all key entities and tasks on top of answering the overall task
2. Verification:
   - Cross-reference data points where possible
   - Validate critical findings
"""
                )
                try:
                    final_response = await self.response_compiler.ainvoke(
                        [
                            SystemMessage(content=self.system_prompt),
                            HumanMessage(content=final_compile_prompt),
                        ]
                    )
                except ValidationError as e:
                    print("Validation error in final response:", e)
                    print("[Retrying.....]")
                    final_response = await self.response_compiler.ainvoke(
                        [
                            SystemMessage(content=self.system_prompt),
                            HumanMessage(content=final_compile_prompt),
                        ]
                    )
            else:
                print(
                    f"\n[{self.name}] No deficiencies found - using initial compilation"
                )
                final_response = initial_response

            # Record in TaskFlow
            self.manager.current_task_flow.set_specialist_response(
                self.name,
                final_response.Final_Response,
                final_response.Plottable_Information,
            )

            # Return in format expected by graph
            return {
                "specialist_responses": {
                    f"{self.name}_response": final_response.Final_Response,
                    f"{self.name}_plottable": final_response.Plottable_Information,
                }
            }

        except Exception as e:
            print(f"[{self.name}] ❌ Error during analysis: {str(e)}")
            print(f"[{self.name}] Full error traceback:")
            traceback.print_exc()
            raise


# Main manager class that coordinates specialists and workflow
class SQUIDManager:
    """
    Main coordinator class that manages specialists and orchestrates the analysis workflow.
    
    Handles initialization of tools, specialists, and coordinates the planning and 
    execution of complex analysis tasks across multiple specialist agents.
    
    Attributes:
        use_internet (bool): Whether internet-based tools are enabled
        agent_capabilities (str): Description of available agent capabilities
        jwt_token (str): Authentication token
        query (str): Original user query
        temp_files (List[str]): Temporary file paths
        selected_docs (List[str]): Selected document references
        filter (Any): Query filter
        relevant_docs (dict): Mapping of relevant documents
        prompt_reg (PromptStore): Prompt template registry
        llm (ChatOpenAI): Main language model
        compile_llm (ChatOpenAI): Model for response compilation
        plot_compile_llm (ChatOpenAI): Model for plot compilation
        specialists (dict): Map of specialist instances
        current_task_flow (TaskFlow): Current task execution tracker
        planner (ThoughtBasedPlanner): Task planning component
    """

    def __init__(
        self,
        llm_name: str = "gpt-4o-mini",
        use_internet: bool = True,
        jwt_token: str = "",
        selected_docs: List[str] = [],
        query: str = "",
        temp_files: List[str] = [],
    ):
        self.use_internet = use_internet
        if self.use_internet:
            self.agent_capabilities = AGENT_CAPABILITIES
        else:
            self.agent_capabilities = AGENT_BASIC_CAPABILITIES
        self.jwt_token = jwt_token
        self.query = query
        self.temp_files = temp_files
        self.selected_docs = selected_docs
        self.filter, self.relevant_docs = MetadataGenerator(
            jwt_token=jwt_token, selected_docs=self.selected_docs
        ).get_filter_and_relevant_docs(query=query)

        self.prompt_reg = PromptStore(
            docs=(
                "\n".join(
                    [
                        f"- {key}: Key Entities: {', '.join(value)}"
                        for key, value in self.relevant_docs.items()
                    ]
                )
                if self.relevant_docs
                else "No relevant docs found"
            ),
            agent_capabilities=self.agent_capabilities,
        )

        self.llm = ChatOpenAI(
            model=llm_name, temperature=0, api_key=config.OPENAI_API_KEY
        )
        self.compile_llm = ChatOpenAI(
            model="gpt-4o", temperature=0, api_key=config.OPENAI_API_KEY
        )
        self.plot_compile_llm = ChatOpenAI(
            model="gpt-4o", temperature=0, api_key=config.OPENAI_API_KEY
        )
        self.compliance_tool = ComplianceChecker(
            llm=self.llm, temp_files=self.temp_files
        ).get_compliance_tool()
        self.ipr_tool = IPRAgent(
            llm=self.llm, temp_files=self.temp_files
        ).get_ipr_analysis_tool()
        self.tools = self._initialize_tools()
        self.specialists = self._initialize_specialists()
        self.current_task_flow = None
        # Add new planner initialization
        self.planner = ThoughtBasedPlanner(self.llm, self.prompt_reg)

        # Track critical tools that need special handling
        self.critical_tools = {
            "multi_hop_tool": self.multi_hop_tool,
            "yahoo_finance_news": self.yahoo_finance_news,
            "tavily": self.tavily,
        }

        # Map of available tools for each specialist
        self.available_tools = {
            "stock_analyst": [tool.name for tool in self.tools["stock_analyst"]],
            "economic_specialist": [
                tool.name for tool in self.tools["economic_specialist"]
            ],
            "market_specialist": [
                tool.name for tool in self.tools["market_specialist"]
            ],
            "compliance_specialist": [
                tool.name for tool in self.tools["compliance_specialist"]
            ],
            "legal_researcher": [tool.name for tool in self.tools["legal_researcher"]],
            "contract_analyst": [tool.name for tool in self.tools["contract_analyst"]],
            "generalist": [tool.name for tool in self.tools["generalist"]],
        }

    """
    Args:
        None
    Returns:
        dict: Map of tool sets for each specialist type
    """
    def _initialize_tools(self) -> dict:
        """Initialize tool sets for different specialists"""
        # Initialize basic tools
        self.tavily = TavilySearchResults(maxResults=5, search_depth="advanced")
        self.jina = JinaSearchTool().get_tool()
        self.serper = SerperSearchTool().get_tool()
        self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.yahoo_finance_news = YahooFinanceNewsTool()
        self.multi_hop_tool = MultiHopTool(
            llm="gpt-4o-mini",
            filter=self.filter,
            query=self.query,
        )
        self.pdf_query_tool = PDFQueryTool()
        self.graph_tool = GraphTool(llm=self.llm)

        self.polygon_tools = poly_tools
        self.findata_tools = findata_tools
        self.yf_tools = yf_tools

        self.dcf = dcf_tool.DCFAnalysisTool()

        self.legal_analyst = LegalAnalyst(self.llm).get_legal_tool()
        self.macroeconomics_tool = MacroeconomicAnalysisAgent(
            self.llm
        ).get_macroeconomic_tool()
        self.corporateactions_tool = CorporateActionsAgent(
            self.llm
        ).get_corporate_actions_tool()
        self.financialstatement_tool = FinancialStatementAnalysisAgent(
            self.llm
        ).get_financial_statement_tool()
        self.financialnews_tool = FinancialNewsAgent(self.llm).get_news_tool()
        # self.valuation_tool = ValuationTool(self.llm).get_valuation_tool()
        self.technical_indicator_tool = TechnicalIndicatorAnalyst(
            self.llm
        ).get_technical_tool()

        # Add common tools to each specialist
        common_tools = [
            self.tavily,
            self.multi_hop_tool,
            self.serper,
            self.jina,
            self.pdf_query_tool,
        ]

        if self.use_internet:
            return {
                "stock_analyst": [
                    *common_tools,
                    self.yahoo_finance_news,
                    *self.polygon_tools,
                    self.dcf,
                    *self.findata_tools,
                    self.corporateactions_tool,
                    self.financialstatement_tool,
                    self.financialnews_tool,
                ],
                "economic_specialist": [
                    *common_tools,
                    self.wikipedia,
                    self.yahoo_finance_news,
                    *self.polygon_tools,
                    *self.findata_tools,
                    self.macroeconomics_tool,
                    self.financialstatement_tool,
                ],
                "market_specialist": [
                    *common_tools,
                    self.wikipedia,
                    self.yahoo_finance_news,
                    self.dcf,
                    self.technical_indicator_tool,
                    self.corporateactions_tool,
                    self.financialnews_tool,
                ],
                "generalist": [*common_tools, self.wikipedia],
                "legal_researcher": [
                    *common_tools,
                    self.wikipedia,
                    self.compliance_tool,
                    self.ipr_tool,
                    self.legal_analyst,
                ],
                "contract_analyst": [
                    *common_tools,
                    self.wikipedia,
                    self.corporateactions_tool,
                    self.compliance_tool,
                    self.ipr_tool,
                    self.legal_analyst,
                ],
                "compliance_specialist": [
                    *common_tools,
                    self.wikipedia,
                    self.compliance_tool,
                    self.ipr_tool,
                    self.legal_analyst,
                ],
            }

        else:
            return {
                "stock_analyst": [
                    self.multi_hop_tool,
                    self.pdf_query_tool,
                ],
                "economic_specialist": [
                    self.multi_hop_tool,
                    self.pdf_query_tool,
                ],
                "market_specialist": [
                    self.multi_hop_tool,
                    self.pdf_query_tool,
                ],
                "generalist": [self.multi_hop_tool, self.pdf_query_tool],
                "legal_researcher": [
                    self.multi_hop_tool,
                    self.legal_analyst,
                    self.pdf_query_tool,
                    self.ipr_tool,
                ],
                "contract_analyst": [
                    self.multi_hop_tool,
                    self.legal_analyst,
                    self.pdf_query_tool,
                    self.ipr_tool,
                ],
                "compliance_specialist": [
                    self.multi_hop_tool,
                    self.legal_analyst,
                    self.pdf_query_tool,
                    self.compliance_tool,
                    self.ipr_tool,
                ],
            }

    """
    Args:
        None
    Returns:
        dict: Map of initialized specialist instances
    """
    def _initialize_specialists(self) -> dict:
        """Initialize specialist instances with their respective tools"""
        tools = self._initialize_tools()
        return {
            name: SQUIDSpecialist(name, self.llm, tools[name], self)
            for name in tools.keys()
        }

    def _validate_task_structure(self, task_data: Dict[str, Any]) -> bool:
        required_fields = {"task_description", "key_entities", "key_tasks"}
        return all(field in task_data for field in required_fields)

    async def unified_planning(self, state: State) -> dict:
        # Generate and validate execution plan:
        # 1. Create task assignments
        # 2. Select specialists
        # 3. Set execution parameters
        """Single planning node that replaces both selection and delegation"""

        query = state["messages"][-1].content

        try:
            plan = await self.planner.generate_plan(
                query=query,
            )

            # Validate plan structure before proceeding
            if not plan.tasks:
                raise ValueError("Generated plan has no tasks")

            if not all(spec in plan.tasks for spec in plan.selected_specialists):
                raise ValueError(
                    f"Plan missing tasks for some specialists: {set(plan.selected_specialists) - set(plan.tasks.keys())}"
                )

            # Create task flow tracking
            if self.current_task_flow is None:
                self.current_task_flow = TaskFlow(original_query=query)

            # Structure tasks for each selected specialist
            structured_tasks = {}
            for specialist in plan.selected_specialists:
                if specialist in plan.tasks:
                    task_data = plan.tasks[specialist]
                    if self._validate_task_structure(task_data):
                        structured_tasks[specialist] = task_data
                        # Record in task flow
                        self.current_task_flow.add_specialist_task(
                            specialist_name=specialist,
                            task=plan.tasks[specialist]["task_description"],
                            iteration_mode=1,
                        )
                    else:
                        raise ValueError(f"Invalid task structure for {specialist}")

            print(f"\nPlanning complete:")
            print(f"Selected specialists: {plan.selected_specialists}")
            print(f"Query complexity: {plan.complexity}")
            print(f"Response length: {plan.response_length}")
            print(f"Iteration modes: 1")

            # Return state updates
            return {
                "selected_agents": plan.selected_specialists,
                "iteration_modes": 1,
                "current_tasks": [structured_tasks],
                "query_complexity": plan.complexity,
                "response_length": plan.response_length,
                "iteration_state": {spec: 1 for spec in plan.selected_specialists},
            }

        except Exception as e:
            print(f"Error in unified planning: {str(e)}")
            traceback.print_exc()
            raise

    async def compile_analysis(self, state: State) -> dict:
        # Combine specialist responses:
        # 1. Gather all responses
        # 2. Generate plots if needed
        # 3. Create final compilation
        """Compile analysis from all specialists' responses"""
        responses = state["specialist_responses"]
        tasks_dict = state["current_tasks"][-1] if state["current_tasks"] else {}
        selected_agents = state.get("selected_agents", [])
        original_query = state["messages"][0].content

        # Validate we have responses to compile
        if not responses:
            print("WARNING: No specialist responses available for compilation!")
            print(f"Current state: {state}")
            raise ValueError("Cannot compile: No specialist responses available")

        # Check if all specialists have completed
        pending_specialists = []
        for agent in selected_agents:
            if f"{agent}_response" not in responses:
                pending_specialists.append(agent)

        if pending_specialists:
            print(f"WARNING: Waiting for responses from: {pending_specialists}")
            print(f"Available responses: {list(responses.keys())}")
            return {"messages": ["Awaiting specialist completion"]}

        compilation_prompt = f"""
        Synthesize a comprehensive analysis for this query:
        Original Query: {original_query}

        Here are the specialist analyses from the selected experts:
        """

        plot_compilation_prompt = f"""
        Synthesize a comprehensive plot information for this query:
        Original Query: {original_query}

        Here are the specialist plottable informations from the selected experts:
        """

        for agent in selected_agents:
            # Use task information from both task_dict and specialist_tasks
            agent_task = tasks_dict.get(agent, "")
            if agent in self.current_task_flow.specialist_tasks:
                specialist_task = self.current_task_flow.specialist_tasks[agent]

                compilation_prompt += f"\n{'='*20}\n"
                compilation_prompt += f"{agent.replace('_', ' ').title()}:\n"
                compilation_prompt += f"Main Task: {specialist_task.task_description}\n"
                compilation_prompt += f"Assigned Task Details: {agent_task}\n"
                compilation_prompt += "Key Entities Analyzed:\n"
                compilation_prompt += "\n".join(
                    f"- {entity}" for entity in specialist_task.key_entities
                )

                # Get response and plottable info
                response_key = f"{agent}_response"
                plottable_key = f"{agent}_plottable"

                analysis = responses.get(response_key)
                if not analysis:
                    print(f"WARNING: No analysis found for {agent}")
                    continue

                plottable_info = responses.get(plottable_key, "None")

                compilation_prompt += f"\nAnalysis: {analysis}\n"
                plot_compilation_prompt += f"Plottable Information: {plottable_info}\n"
                compilation_prompt += f"{'='*20}\n"
                plot_compilation_prompt += f"{'='*20}\n"

        # Add compilation guidelines based on complexity
        compilation_prompt += self.prompt_reg._return_response_compiling_prompt(
            task_complexity=state["query_complexity"],
            response_length=state["response_length"],
        )
        plot_compilation_prompt += self.prompt_reg._return_plot_compiling_prompt(
            task_complexity=state["query_complexity"],
            response_length=state["response_length"],
        )
        print("plot_compilation prompt \n", plot_compilation_prompt)
        print("[MANAGER] Starting final compilation...")
        print(f"[MANAGER] Compiling responses from {len(responses)} specialists")

        try:
            final_response = await self.compile_llm.ainvoke(
                [HumanMessage(content=compilation_prompt)]
            )
            plot_final_response = await self.plot_compile_llm.ainvoke(
                [HumanMessage(content=plot_compilation_prompt)]
            )
            print("*" * 50)
            final_response_text = final_response.content
            plottable_info = plot_final_response.content

            print("Plottable info found is:")
            print(plottable_info)

            print(f"Final Response Generated")
            print(f"Plottable Information: {plottable_info}")

            # Handle plot generation if applicable
            if plottable_info and plottable_info.lower() != "none":
                try:
                    plot = self.graph_tool._run(query=plottable_info)
                    print(f"Plot generation result: {plot}")

                    if plot.get("metadata") and plot["metadata"].get("image_path"):
                        img_path = plot["metadata"]["image_path"]
                        if os.path.exists(img_path):
                            self.current_task_flow.set_image_path(img_path)
                            print(f"Successfully saved plot to: {img_path}")
                        else:
                            print(
                                f"Warning: Generated image path does not exist: {img_path}"
                            )
                except Exception as plot_error:
                    print(f"Warning: Error during plot generation: {str(plot_error)}")
            else:
                print("No plottable information provided")

            # Update task flow and return
            if self.current_task_flow:
                self.current_task_flow.set_final_compilation(final_response_text)

            print(f"\nCompilation completed with {len(selected_agents)} specialists")
            print(f"Available responses: {list(responses.keys())}")

            return {"messages": [final_response_text]}

        except Exception as e:
            print(f"Error during compilation: {str(e)}")
            traceback.print_exc()
            raise
