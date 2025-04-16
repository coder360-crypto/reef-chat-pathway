import os
import platform
import sys
from datetime import datetime

import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from config import AgentsConfig as Config
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from utils.moray_utils.joiner import Joiner
from utils.moray_utils.planner import Planner
from utils.moray_utils.scheduler import Scheduler
from utils.moray_utils.state_manager import StateManager
from utils.moray_utils.tool_manager import ToolManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    """State that tracks messages"""

    messages: Annotated[List[BaseMessage], add_messages]


class MORAY:
    """
    Main MORAY service for processing and executing user queries.

    This class orchestrates:
    - Language model selection and configuration
    - Tool initialization and management
    - Query processing and execution
    - Response generation and formatting

    Attributes:
        _tool_cache (Dict[str, Dict[str, Any]]): Class-level cache for reusable tools
        config (Config): Configuration settings
        llm (BaseChatModel): Language model instance
        attempt_count (int): Number of processing attempts
        tool_manager (ToolManager): Manager for available tools
    """

    _tool_cache: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        """Initialize the MORAY instance with default settings."""

        # Load configuration
        self.config = Config()

        # Initialize language model
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=self.config.OPENAI_API_KEY,
        )

        # Initialize state
        self.attempt_count = 0
        self.inputs_url = self.config.INPUTS_URL

        # Initialize components
        self.tool_manager = ToolManager()
        self.execution_metrics = {}

    def process_user_query(
        self,
        query: str,
        context: str = None,
        llm: str = None,
        temp_files: List[str] = None,
        selected_docs: List[str] = None,
        jwt_token: str = None,
        use_internet: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """Process a user query and generate a response with execution metrics."""
        execution_start = datetime.now()
        callbacks = []
        outputs = []
        self.context = context
        self.query = query

        try:
            with get_openai_callback() as cb:
                # Store query parameters
                self.selected_docs = selected_docs
                self.jwt_token = jwt_token
                self.temp_files = temp_files if temp_files else []
                self.attempt_count = 0

                # Configure language model if specified
                if llm is not None:
                    self._configure_language_model(llm)

                # Initialize tools and descriptions
                self._initialize_tools(use_internet)

                # Set up processing components
                planner = Planner(self.llm, self.tool_manager)
                scheduler = Scheduler()
                self.joiner = Joiner(self.llm, self.query)
                self.state_manager = StateManager(scheduler, planner, self.joiner)

                # Create and execute processing graph
                execution_graph = self.state_manager.get_execution_graph()
                chain = execution_graph.compile()

                # Prepare initial state
                initial_state = self._create_initial_state()

                # Process query and collect outputs with metrics
                for step in chain.stream(initial_state):
                    outputs.append(step)
                    step_time = datetime.now()
                    step_name = (
                        list(step.keys())[0]
                        if isinstance(step, dict)
                        else "process_step"
                    )

                    # Log step details
                    logger.info(f"Step completed: {step_name}")
                    logger.info(f"Time: {step_time.strftime('%H:%M:%S')}")

                    if "messages" in step:
                        last_message = step["messages"][-1]
                        if hasattr(last_message, "content"):
                            preview = last_message.content[:150] + "..."
                            logger.info(f"Preview: {preview}")

                    callback_info = {
                        "step": step_name,
                        "timestamp": step_time.isoformat(),
                        "total_tokens": cb.total_tokens,
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_cost": cb.total_cost,
                        "execution_time": (step_time - execution_start).total_seconds(),
                    }
                    callbacks.append(callback_info)

                    logger.info(
                        f"Tokens used in step: {callback_info['completion_tokens']}"
                    )
                    logger.info(f"Running cost: ${callback_info['total_cost']:.4f}")

        except Exception as e:
            error_time = datetime.now()
            logger.error(
                f"Error during execution at {error_time.strftime('%H:%M:%S')}: {str(e)}"
            )
            callbacks.append(
                {
                    "step": "error",
                    "timestamp": error_time.isoformat(),
                    "error": str(e),
                    "execution_time": (error_time - execution_start).total_seconds(),
                    "total_tokens": cb.total_tokens if "cb" in locals() else 0,
                    "total_cost": cb.total_cost if "cb" in locals() else 0,
                }
            )
            raise

        finally:
            execution_time = datetime.now() - execution_start
            logger.info(f"Total execution time: {execution_time}")

            # Get the last valid callback or use defaults
            last_callback = next(
                (cb for cb in reversed(callbacks) if "total_tokens" in cb),
                {"total_tokens": 0, "total_cost": 0.0},
            )

            self.execution_metrics = {
                "query": query,
                "context": context,
                "execution_time": str(execution_time),
                "callbacks": callbacks,
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "platform": platform.system(),
                    "python_version": platform.python_version(),
                    "memory_usage": psutil.Process().memory_info().rss
                    / 1024
                    / 1024,  # MB
                },
                "total_metrics": {
                    "total_time": str(execution_time),
                    "total_steps": len(outputs),
                    "input_tokens": last_callback.get("prompt_tokens", 0),
                    "output_tokens": last_callback.get("completion_tokens", 0),
                    "total_tokens": last_callback.get("total_tokens", 0),
                    "total_cost": last_callback.get("total_cost", 0.0),
                },
            }

        return {
            "results": outputs,
            "callbacks": callbacks,
            "execution_metrics": self.execution_metrics,
        }

    def _configure_language_model(self, llm: str):
        """
        Configure the language model based on type.

        Args:
            llm: Language model identifier
        """
        if "gpt" in llm.lower():
            self.llm = ChatOpenAI(
                model=llm, temperature=0, api_key=self.config.OPENAI_API_KEY
            )
        else:
            self.llm = ChatGroq(
                model=llm,
                temperature=0,
                api_key=self.config.GROQ_API_KEY,
                max_tokens=8000,
            )

    def _initialize_tools(self, use_internet: bool):
        """
        Initialize tools and set custom descriptions.

        Args:
            query: User's input query
            use_internet: Whether to allow internet access
        """
        self.tool_manager.initialize_tools(
            llm=self.llm,
            query=self.query,
            context=self.context,
            selected_docs=self.selected_docs,
            temp_files=self.temp_files,
            jwt_token=self.jwt_token,
            use_internet=use_internet,
        )

    def _create_initial_state(self) -> dict:
        """
        Create the initial state for query processing.

        Args:
            query: User's input query
            context: Additional context

        Returns:
            Dict containing initial message state
        """
        return {
            "messages": [
                HumanMessage(self.query),
                SystemMessage(f"Context: {self.context}"),
            ]
        }


if __name__ == "__main__":

    print("Starting MORAY service")

    agent = MORAY()

    print("Test query: What is the total revenue of AMCOR in FY2023?")
    response = agent.process_user_query(
        "What is the total revenue of AMCOR in FY2023?", jwt_token=Config().JWT_TOKEN
    )
    print(response)
