import logging
import os
import sys
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, FunctionMessage, SystemMessage
from langchain_core.runnables import RunnableBranch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.moray_utils.output_parser import MORAYPlanParser
from utils.moray_utils.tool_manager import ToolManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from utils.moray_utils.prompts import planner_template, replanner_template

logger = logging.getLogger(__name__)


class PromptConfig:
    replan: str
    instructions: str
    num_tools: int
    tool_descriptions: str


class Planner:
    """
    Manages the planning and replanning of task execution sequences.

    This class handles:
    - Initial task planning based on user input
    - Replanning when initial approaches fail
    - Tracking failed approaches and previous plans
    - Managing tool selection and execution order

    Attributes:
        MAX_REPLANS (int): Maximum number of replanning attempts allowed
        replan_count (int): Current number of replanning attempts
        previous_plans (List): History of executed plans
        failed_approaches (Set): Set of approaches that have failed
        failed_summary (str): Summary of failed attempts
        next_task_index (int): Index for the next task to be executed
    """

    MAX_REPLANS = 3

    def __init__(self, llm: BaseChatModel, tool_manager: ToolManager):
        """
        Initialize the Planner with language model and tools.

        Args:
            llm: Language model for generating plans
            tool_manager: Manager containing available tools
        """
        self.llm = llm
        self.tool_manager = tool_manager

        # Get tool configurations from manager
        self.planner_tool_names = self.tool_manager.planner_tool_names
        self.replanner_tool_names = self.tool_manager.replanner_tool_names
        self.planner_tool_descriptions = self.tool_manager.planner_descriptions
        self.replanner_tool_descriptions = self.tool_manager.replanner_descriptions
        self.tools = self.tool_manager.tools

        # Initialize planning state
        self.replan_count = 0
        self.previous_plans = []
        self.failed_approaches = set()
        self.failed_summary = ""
        self.next_task_index = 1

        # Create prompts and planning chain
        self.planner_prompt = self._create_planner_prompt()
        self.replanner_prompt = self._create_replanner_prompt()
        self.plan = self._create_planning_chain()
    def _create_planning_chain(self):
        """Create the planning chain with branching logic for replanning."""
        return (
            RunnableBranch(
                (
                    self.should_replan,
                    self.wrap_and_get_last_index | self.replanner_prompt,
                ),
                self.wrap_messages | self.planner_prompt,
            )
            | self.llm
            | MORAYPlanParser(tools=self.tools)
        )

    def get_tasks(self, messages: List[BaseMessage]):
        """
        Generate tasks based on input messages.

        Args:
            messages: List of input messages to process

        Returns:
            Generator yielding planned tasks
        """
        return self.plan.stream(messages)

    def should_replan(self, state: list) -> bool:
        """
        Determine if replanning is needed based on execution state.

        Args:
            state: Current execution state

        Returns:
            bool: True if replanning is needed, False otherwise
        """
        if not state or len(state) < 3:
            return False

        # Check if feedback indicates need for replanning
        feedback_message = state[-2]
        needs_replan = (
            isinstance(feedback_message, SystemMessage)
            and "Feedback" in feedback_message.content
        )

        if needs_replan:
            # Extract failed approaches from current attempt
            failed_approaches = set()
            for message in state:
                if isinstance(message, FunctionMessage) and message.name != "join":
                    failed_tool = message.name
                    failed_args = message.additional_kwargs.get("args", "")
                    failed_approaches.add(f"{failed_tool}: {failed_args}")

            # Update feedback with failed approaches
            if len(state) >= 2 and isinstance(state[-2], SystemMessage):
                state[-2].content += "\n\nFailed Approaches:\n"
                for approach in failed_approaches:
                    state[-2].content += f"- {approach}\n"

            # Store failed approaches for future reference
            self.failed_approaches.update(failed_approaches)

        return needs_replan

    def _create_planner_prompt(self):
        """Create the initial planning prompt with tool descriptions."""
        return planner_template.partial(
            num_tools=len(self.planner_tool_names) + 1,  # Add one for join() tool
            tool_descriptions=self.planner_tool_descriptions,
            instructions=(
                f"Use only the following tools to complete tasks: {self.planner_tool_names}. "
                "Do not attempt to use any tools not explicitly listed."
            ),
        )

    def _create_replanner_prompt(self):
        """Create the replanning prompt with failure information."""
        return replanner_template.partial(
            num_tools=len(self.replanner_tool_names) + 1,
            tool_descriptions=self.replanner_tool_descriptions,
            attempt_count=self.replan_count,
            max_attempts=self.MAX_REPLANS + 1,
            failed_approaches="\n".join(self.failed_approaches),
            failed_summary=self.failed_summary,
        )

    def _extract_plan_from_state(self, state: list) -> List[dict]:
        """
        Extract the executed plan from state messages.

        Args:
            state: Current execution state

        Returns:
            List of executed tasks with their results
        """
        plan = []
        for message in state:
            if isinstance(message, FunctionMessage):
                plan.append(
                    {
                        "tool": message.name,
                        "args": message.additional_kwargs.get("args"),
                        "result": message.content,
                        "idx": message.additional_kwargs.get("idx"),
                    }
                )
        return plan

    def wrap_messages(self, state: list) -> dict:
        """Wrap state messages for prompt input."""
        return {"messages": state}

    def wrap_and_get_last_index(self, state: list) -> dict:
        """
        Wrap state with previous plan information.

        Args:
            state: Current execution state

        Returns:
            Dict containing wrapped messages with plan history
        """
        # Extract and store current plan
        current_plan = self._extract_plan_from_state(state)
        self.previous_plans.append(current_plan)

        # Update next task index
        if current_plan:
            self.next_task_index = max(step["idx"] for step in current_plan) + 1

        # Create detailed summary of previous attempts
        summary = self._create_attempt_summary()
        self.failed_summary = self.failed_summary.join(summary)

        # Update the last message with comprehensive context
        if len(state) >= 2 and isinstance(state[-2], SystemMessage):
            state[-2].content = (
                f"{state[-2].content}\n\n"
                f"Next task index: {self.next_task_index}\n"
                f"{summary}"
            )

        return {"messages": state}

    def _create_attempt_summary(self) -> str:
        """Create a detailed summary of previous planning attempts."""
        summary = "\nPrevious Attempts Analysis:\n"

        # Summarize each attempt
        for i, plan in enumerate(self.previous_plans):
            summary += f"\nAttempt {i+1}:\n"
            for step in plan:
                summary += (
                    f"- Used {step['tool']} (Task {step['idx']})\n"
                    f"  Args: {step['args']}\n"
                    f"  Result: {step['result'][:1000]}...\n"
                )

        # Add failure analysis if any
        if self.failed_approaches:
            summary += "\nFailed Approaches:\n"
            for approach in self.failed_approaches:
                summary += f"- {approach}\n"

        return summary


if __name__ == "__main__":
    import time

    from config import AgentsConfig as Config
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from tqdm import tqdm
    from utils.moray_utils.tool_manager import ToolManager

    print("Testing Planner latency...")

    # Initialize components
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=Config().OPENAI_API_KEY)
    tool_manager = ToolManager()
    tool_manager.initialize_tools(
        llm,
        query="What is the total revenue of AMCOR in FY2023?",
        jwt_token=Config().JWT_TOKEN,
        use_internet=True,
        temp_files=["nda.pdf"],
        selected_docs=[],
    )
    planner = Planner(llm, tool_manager)

    test_queries = [
        {
            "messages": [
                SystemMessage(
                    content="Planning task execution based on the following query:"
                ),
                HumanMessage(
                    content="Give me the balance sheet and income statement of Tesla"
                ),
            ]
        }
    ]

    # Test latency
    total_time = 0

    for state in tqdm(test_queries, desc="Running tests"):
        start_time = time.time()

        # Process using the planner
        tasks = planner.get_tasks(state["messages"])
        for task in tasks:
            pass

        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"\nTest completion time: {elapsed:.2f} seconds")
        # Optionally print task details or results

    avg_time = total_time / (len(test_queries))
    print(f"\nAverage processing time: {avg_time:.2f} seconds per query")
