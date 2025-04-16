import itertools
import logging
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from utils.moray_utils.joiner import Joiner
from utils.moray_utils.planner import Planner
from utils.moray_utils.scheduler import Scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class State(TypedDict):
    """State dictionary for tracking messages in the execution graph."""

    messages: Annotated[List[BaseMessage], add_messages]


def get_plan_and_schedule_chain(scheduler: Scheduler, planner: Planner):
    @as_runnable
    def plan_and_schedule(state: State, config: RunnableConfig) -> dict:
        """Execute planning and scheduling of tasks based on the current state.

        Args:
            state: Current state containing messages
            config: Runnable configuration

        Returns:
            dict: Updated state with scheduled tasks
        """
        messages = state["messages"]
        tasks = planner.get_tasks(messages)

        try:
            # Ensure we can peek at first task without consuming the iterator
            first_task = next(tasks)
            tasks = itertools.chain([first_task], tasks)
            logger.info(f"Generated tasks: {tasks}")
        except StopIteration:
            logger.warning("No tasks were generated")
            tasks = iter([])

        scheduled_tasks = scheduler.schedule_tasks(
            {"messages": messages, "tasks": tasks},
            config,
        )
        logger.info(f"Scheduled tasks: {scheduled_tasks}")
        return {"messages": scheduled_tasks}

    return plan_and_schedule


class StateManager:
    """Manages the execution state and graph for the task processing pipeline."""

    def __init__(self, scheduler: Scheduler, planner: Planner, joiner: Joiner):
        self.scheduler = scheduler
        self.planner = planner
        self.joiner = joiner
        self.attempt_count = 0

    def should_continue(self, state: State) -> str:
        """Determine if the execution should continue or end based on current state.

        Args:
            state: Current execution state

        Returns:
            str: Next node to execute or END
        """
        messages = state["messages"]
        last_message = messages[-2]

        if isinstance(last_message, AIMessage):
            return END

        needs_replan = isinstance(last_message, SystemMessage)
        if needs_replan:
            self.attempt_count += 1

            if self.attempt_count >= Planner.MAX_REPLANS:
                logger.warning(
                    f"Maximum attempts ({Planner.MAX_REPLANS}) reached. Using last complete response."
                )
                return END

            logger.info(f"Starting attempt {self.attempt_count}/{Planner.MAX_REPLANS}")
            return "plan_and_schedule"

        return END

    def get_execution_graph(self) -> StateGraph:
        """Create and configure the execution graph.

        Returns:
            StateGraph: Configured execution graph
        """
        graph_builder = StateGraph(State)

        # Add nodes and edges
        graph_builder.add_node(
            "plan_and_schedule",
            get_plan_and_schedule_chain(self.scheduler, self.planner),
        )
        graph_builder.add_node("join", self.joiner.get_chain())

        graph_builder.add_edge("plan_and_schedule", "join")
        graph_builder.add_edge(START, "plan_and_schedule")
        graph_builder.add_conditional_edges("join", self.should_continue)

        return graph_builder
