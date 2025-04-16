import asyncio
import json
import logging
import os
import platform
import sys
from datetime import datetime, timedelta

import psutil

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from typing import Any, Dict, List

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from utils.squid_utils.squid_manager import SQUIDManager, State


def create_specialist_router(spec_name: str):
    """Create routing function for a specialist
    
    Args:
        spec_name (str): Name of the specialist agent
        
    Returns:
        Callable[[State], str]: A routing function that returns either the specialist name or "compile"
    """

    def route_specialist(state: State) -> str:
        # Controls the flow between specialist iterations and compilation
        iteration_mode = 1
        current_iteration = state.get("iteration_state", {}).get(spec_name, 1)

        # Route to second iteration if needed, otherwise proceed to compilation
        if iteration_mode == 2 and current_iteration == 1:
            print(f"[{spec_name}] Routing to second iteration")
            return spec_name

        print(f"[{spec_name}] Routing to compilation")
        return "compile"

    return route_specialist


async def create_analysis_workflow(
    llm_name: str = "gpt-4o-mini",
    use_internet: bool = True,
    jwt_token: str = "",
    selected_docs: List[str] = [],
    query: str = "",
    temp_files: List[str] = [],
) -> tuple:
    """Create the async workflow graph with unified planning
    
    Args:
        llm_name (str): Name of the language model to use
        use_internet (bool): Whether to allow internet access
        jwt_token (str): Authentication token
        selected_docs (List[str]): List of selected document paths
        query (str): User query string
        temp_files (List[str]): List of temporary file paths
        
    Returns:
        Tuple[AsyncGraph, SQUIDManager]: Compiled workflow graph and SQUID manager instance
    """
    print("Initializing SQUIDManager with provided parameters...")
    manager = SQUIDManager(
        llm_name=llm_name,
        use_internet=use_internet,
        jwt_token=jwt_token,
        selected_docs=selected_docs,
        query=query,
        temp_files=temp_files,
    )

    print("Creating StateGraph...")
    graph = StateGraph(State)

    # Add unified planning node
    print("Adding unified planning node...")
    graph.add_node("plan", manager.unified_planning)

    # Add specialist nodes
    print("Adding specialist nodes...")
    specialist_nodes = {
        "stock_analyst": manager.specialists["stock_analyst"].analyze,
        "economic_specialist": manager.specialists["economic_specialist"].analyze,
        "market_specialist": manager.specialists["market_specialist"].analyze,
        "compliance_specialist": manager.specialists["compliance_specialist"].analyze,
        "legal_researcher": manager.specialists["legal_researcher"].analyze,
        "contract_analyst": manager.specialists["contract_analyst"].analyze,
        "generalist": manager.specialists["generalist"].analyze,
    }

    for name, node in specialist_nodes.items():
        print(f"Adding node for specialist: {name}")
        graph.add_node(name, node)

    # Add compilation node
    print("Adding compilation node...")
    graph.add_node("compile", manager.compile_analysis)

    # Add initial edge
    print("Adding initial edge from START to plan...")
    graph.add_edge(START, "plan")

    # Define routing logic for determining next processing nodes
    def get_next_nodes(state: State) -> List[str]:
        # Extract routing state variables
        selected = state.get("selected_agents", [])
        iteration_modes = state.get("iteration_modes", {})
        iteration_state = state.get("iteration_state", {})
        specialist_responses = state.get("specialist_responses", {})

        next_nodes = []
        for specialist in selected:
            response_key = f"{specialist}_response"
            # Add specialists that haven't processed or need second iteration
            if response_key not in specialist_responses:
                next_nodes.append(specialist)
            elif (
                iteration_modes.get(specialist) == 2
                and iteration_state.get(specialist) == 1
            ):
                next_nodes.append(specialist)

        # Default to compilation if no specialists need processing
        return next_nodes if next_nodes else ["compile"]

    # Add conditional edges
    print("Adding conditional edges from plan to specialist nodes and compile...")
    all_nodes = list(specialist_nodes.keys()) + ["compile"]
    graph.add_conditional_edges("plan", get_next_nodes, all_nodes)

    # Add specialist routing edges
    for name in specialist_nodes:
        print(f"Adding conditional edges for specialist: {name}")
        graph.add_conditional_edges(
            name, create_specialist_router(name), [name, "compile"]
        )

    # Add final edge
    print("Adding final edge from compile to END...")
    graph.add_edge("compile", END)

    print("Workflow graph creation complete.")
    return graph.compile(), manager


class SQUID:
    """Enhanced SQUID class for query analysis with comprehensive tool support.
    Handles query processing, execution tracking, and result management.
    
    Attributes:
        current_manager (Optional[SQUIDManager]): Current SQUID manager instance
        jwt_token (str): Authentication token
        selected_docs (List[str]): List of selected document paths
        temp_files (List[str]): List of temporary file paths
        query (str): Current user query
    """

    def __init__(self, jwt_token: str = "", selected_docs: List[str] = []):
        self.current_manager = None
        self.jwt_token = jwt_token
        self.selected_docs = selected_docs

    async def process_user_query(
        self,
        query: str,
        context: List[str] = [],
        llm_name: str = "gpt-4o-mini",
        use_internet: bool = True,
        temp_files: List[str] = [],
    ) -> Dict[str, Any]:
        """Asynchronously analyze queries with streaming support and detailed metrics.

        Args:
            query (str): The query to analyze
            context (List[str]): Additional context for the query
            llm_name (str): Name of the LLM to use
            use_internet (bool): Whether to allow internet access
            temp_files (List[str]): List of temporary file paths

        Returns:
            Dict[str, Any]: Dictionary containing:
                - final_results (List): Results from each workflow step
                - callbacks (List[Dict]): Execution metrics for each step
                - task_flow_data (Dict): Task flow information
                - execution_metrics (Dict): Overall execution statistics
                
        Raises:
            Exception: If any error occurs during query processing
        """
        self.temp_files = temp_files
        self.query = query
        # Initialize workflow and manager
        workflow, self.current_manager = await create_analysis_workflow(
            llm_name,
            use_internet,
            self.jwt_token,
            self.selected_docs,
            self.query,
            self.temp_files,
        )

        # Prepare query with context
        if context:
            query = query + "\n\nAdditional Context Given by user:\n" + "".join(context)

        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_tasks": [],
            "specialist_responses": {},
            "selected_agents": [],
            "iteration_modes": {},
            "iteration_state": {},
            "execution_state": {},
            "query_complexity": "",
            "response_length": "",
        }

        # Track execution metrics and results
        results = []
        callbacks = []
        execution_start = datetime.now()

        try:
            with get_openai_callback() as cb:
                # Process each step in the workflow
                async for step in workflow.astream(initial_state, stream_mode="values"):
                    # Record step results and execution metrics
                    results.append(step)
                    step_name = list(step.keys())[0]
                    print(f"\nStep completed: {step_name}")
                    step_time = datetime.now()

                    # Log step details
                    print(f"Time: {step_time.strftime('%H:%M:%S')}")
                    if "messages" in step:
                        last_message = step["messages"][-1]
                        if hasattr(last_message, "content"):
                            preview = last_message.content[:150] + "..."
                            print(f"Preview: {preview}")

                    # Build callback info for monitoring and debugging
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

                    # Print execution metrics
                    print(f"Tokens used in step: {callback_info['completion_tokens']}")
                    print(f"Running cost: ${callback_info['total_cost']:.4f}")

        except Exception as e:
            error_time = datetime.now()
            print(
                f"Error during execution at {error_time.strftime('%H:%M:%S')}: {str(e)}"
            )

            # Record error in callbacks
            callbacks.append(
                {
                    "step": "error",
                    "timestamp": error_time.isoformat(),
                    "error": str(e),
                    "execution_time": (error_time - execution_start).total_seconds(),
                }
            )
            raise

        finally:
            # Save execution data and task flow information
            execution_time = datetime.now() - execution_start
            print(f"\nTotal execution time: {execution_time}")

            # Save execution data and task flow
            try:
                task_flow = self.current_manager.current_task_flow
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save task flow if available
                if task_flow:
                    await self._save_task_flow(task_flow, timestamp)
                    task_flow_data = task_flow.to_dict()
                else:
                    task_flow_data = {"error": "No task flow data available"}

                # Save execution metrics
                await self._save_execution_metrics(
                    timestamp=timestamp,
                    query=query,
                    context=context,
                    execution_time=execution_time,
                    callbacks=callbacks,
                )

            except Exception as e:
                print(f"Warning: Could not save execution data: {e}")
                task_flow_data = {"error": str(e)}

        # Return comprehensive results
        return {
            "final_results": results,
            "callbacks": callbacks,
            "task_flow_data": task_flow_data,
            "execution_metrics": {
                "total_time": str(execution_time),
                "total_steps": len(results),
                "input_tokens": callbacks[-1]["prompt_tokens"] if callbacks else 0,
                "output_tokens": callbacks[-1]["completion_tokens"] if callbacks else 0,
                "total_tokens": callbacks[-1]["total_tokens"] if callbacks else 0,
                "total_cost": callbacks[-1]["total_cost"] if callbacks else 0,
            },
        }

    async def _save_task_flow(self, task_flow, timestamp: str) -> None:
        """
        Asynchronously save task flow data to file.

        Args:
            task_flow (TaskFlow): TaskFlow object to save
            timestamp (str): Timestamp for filename

        Returns:
            None
        """
        filename = f"task_flow.json"
        await asyncio.to_thread(task_flow.save_to_file, filename)
        print(f"Task flow saved to: {filename}")

    async def _save_execution_metrics(
        self,
        timestamp: str,
        query: str,
        context: List[str],
        execution_time: timedelta,
        callbacks: List[Dict],
    ) -> None:
        """Asynchronously save execution metrics to file.

        Args:
            timestamp (str): Timestamp for filename
            query (str): Original query
            context (List[str]): Query context
            execution_time (timedelta): Total execution time
            callbacks (List[Dict]): Execution callbacks

        Returns:
            None
        """
        metrics = {
            "query": query,
            "context": context,
            "execution_time": str(execution_time),
            "callbacks": callbacks,
            "timestamp": timestamp,
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            },
        }

        filename = f"execution_metrics.json"
        async with asyncio.Lock():
            await asyncio.to_thread(self._write_json_file, filename, metrics)
        print(f"Execution metrics saved to: {filename}")

    @staticmethod
    def _write_json_file(filename: str, data: Dict) -> None:
        """Helper method to write JSON file.

        Args:
            filename (str): Output filename
            data (Dict): Data to write

        Returns:
            None
        """
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)


async def main():
    import json
    import time

    from config import AgentsConfig as Config
    from tqdm import tqdm

    agent = SQUID(
        jwt_token=Config().JWT_TOKEN,
        query="What is the total revenue of AMCOR in FY2023?",
    )

    print("Starting squid service")

    print("Test query: What is the total revenue of AMCOR in FY2023?")
    response = await agent.process_user_query(
        "What is the total revenue of AMCOR in FY2023?",
        llm_name="gpt-4o",
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
