import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Set, Union

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.moray_utils.output_parser import Task

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


class Scheduler:
    """
    Manages the execution scheduling of planned tasks.

    This class handles:
    - Task dependency resolution
    - Parallel task execution
    - Task validation
    - Result collection and formatting

    The scheduler ensures tasks are executed in the correct order while maximizing
    parallel execution where possible.
    """

    def __init__(self):
        """Initialize the scheduler with empty task list."""
        self.tasks = None

    def _build_graph(self) -> Dict[int, Set[int]]:
        """
        Build a dependency graph from tasks.

        Returns:
            Dict mapping task indices to sets of their dependencies
        """
        graph = {}
        for task in self.tasks:
            # Initialize task node with empty dependency set
            graph.setdefault(task["idx"], set())
            for dep in task["dependencies"]:
                # Add dependencies to graph
                graph.setdefault(dep, set())
                graph[task["idx"]].add(dep)
        return graph

    def _has_cycle(
        self,
        graph: Dict[int, Set[int]],
        node: int,
        visited: Dict[int, bool],
        path: Dict[int, bool],
    ) -> bool:
        """
        Check for cycles in the dependency graph using DFS.

        Args:
            graph: Dependency graph to check
            node: Current node being visited
            visited: Nodes visited in entire DFS
            path: Nodes visited in current path

        Returns:
            bool: True if cycle detected, False otherwise
        """
        if node not in graph:
            return False

        visited[node] = True
        path[node] = True

        for neighbor in graph[node]:
            if neighbor not in visited:
                if self._has_cycle(graph, neighbor, visited, path):
                    return True
            elif path[neighbor]:
                return True

        path[node] = False
        return False

    def _check_circular_dependencies(self) -> None:

        graph = self._build_graph()
        visited = {}
        path = {}

        for node in graph:
            if node not in visited:
                if self._has_cycle(graph, node, visited, path):
                    raise ValueError(f"Circular dependency detected in task graph")

    def _get_observations(self, messages: List[BaseMessage]) -> Dict[int, Any]:
        """
        Extract previous tool responses from messages.

        Args:
            messages: List of messages to process

        Returns:
            Dict mapping task indices to their results
        """
        results = {}
        for message in messages[::-1]:
            if isinstance(message, FunctionMessage):
                results[int(message.additional_kwargs["idx"])] = message.content
        return results

    def _execute_task(
        self,
        task: Task,
        observations: Dict[int, Any],
        config: Dict[str, Any],
    ) -> Any:
        """
        Execute a single task with resolved arguments.

        Args:
            task: Task to execute
            observations: Previous task results
            config: Execution configuration

        Returns:
            Result of task execution

        Raises:
            Exception: If task execution fails
        """
        tool_to_use = task["tool"]
        if isinstance(tool_to_use, str):
            return tool_to_use

        try:
            args = task["args"]
            resolved_args = self._resolve_task_args(args, observations)
            return tool_to_use.invoke(resolved_args, config)

        except Exception as e:
            return f"ERROR(Failed to call {tool_to_use.name} with args {args}.) Args could not be resolved. Error: {repr(e)}"

    def _resolve_task_args(
        self, args: Union[str, List, Dict], observations: Dict[int, Any]
    ) -> Any:
        """
        Resolve task arguments by replacing references with actual values.
        Handles multiple dependency references in any string format.

        Args:
            args: Arguments to resolve
            observations: Previous task results to use for resolution

        Returns:
            Resolved arguments
        """
        ID_PATTERN = r"\$\{?(\d+)\}?"

        if isinstance(args, str):
            # Find all matches in the string
            matches = list(re.finditer(ID_PATTERN, args))
            if not matches:
                return args

            # Process the string with all matches
            result = args
            for match in matches[::-1]:  # Reverse order to handle overlapping matches
                start, end = match.span()
                idx = int(match.group(1))
                replacement = str(observations.get(idx, match.group(0)))
                result = result[:start] + replacement + result[end:]
            return result

        elif isinstance(args, list):
            return [self._resolve_task_args(a, observations) for a in args]
        elif isinstance(args, dict):
            return {
                key: self._resolve_task_args(value, observations)
                for key, value in args.items()
            }
        else:
            return str(args)

    def schedule_task(
        self, task_inputs: Dict[str, Any], config: RunnableConfig
    ) -> None:
        """
        Schedule and execute a single task.

        Args:
            task_inputs: Task and observation data
            config: Execution configuration
        """
        task: Task = task_inputs["task"]
        observations: Dict[int, Any] = task_inputs["observations"]

        try:
            result = self._execute_task(task, observations, config)
        except Exception:
            error_msg = (
                f"Task {task['idx']} failed with error: {traceback.format_exc()}"
            )
            logging.error(error_msg)
            result = error_msg

        observations[task["idx"]] = result

    def schedule_pending_task(
        self,
        task: Task,
        observations: Dict[int, Any],
        retry_after: float = 0.2,
        timeout: float = 10.0,
    ) -> None:
        """
        Schedule a task that depends on other tasks.

        Args:
            task: Task to schedule
            observations: Current task results
            retry_after: Seconds to wait between dependency checks
            timeout: Maximum seconds to wait for dependencies

        Raises:
            TimeoutError: If dependencies not satisfied within timeout
        """
        start_time = time.time()

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Task {task['idx']} timed out after {timeout} seconds"
                )

            # Check dependencies
            deps = task["dependencies"]
            if deps and any(dep not in observations for dep in deps):
                time.sleep(retry_after)
                continue

            # All dependencies satisfied
            logger.info("Scheduling task: %s with observations: %s", task, observations)
            self.schedule_task({"task": task, "observations": observations})
            break

    def _validate_tasks(self) -> None:
        """
        Validate task structure and check for circular dependencies.

        Raises:
            ValueError: If tasks are invalid or contain circular dependencies
        """
        required_fields = ["idx", "tool", "args", "dependencies"]

        # Validate task structure
        for task in self.tasks:
            missing = [field for field in required_fields if field not in task]
            if missing:
                raise ValueError(f"Task missing required fields: {missing}")

            if not isinstance(task["dependencies"], (list, tuple, set)):
                raise ValueError(
                    f"Task dependencies must be iterable, got {type(task['dependencies'])}"
                )

        # Check for circular dependencies
        self._check_circular_dependencies()

    def schedule_tasks(
        self, scheduler_input: SchedulerInput, config: RunnableConfig
    ) -> List[FunctionMessage]:
        """
        Schedule and execute tasks based on their dependencies.

        Args:
            scheduler_input: Input containing messages and tasks
            config: Configuration for task execution

        Returns:
            List[FunctionMessage]: Messages containing task results

        Raises:
            ValueError: If task validation fails
        """
        self.tasks = list(scheduler_input["tasks"])

        if not self.tasks:
            return []

        try:
            # Validate task structure and dependencies
            self._validate_tasks()

            # Initialize tracking structures
            args_for_tasks = {}
            messages = scheduler_input["messages"]
            observations = self._get_observations(messages)
            task_names = {}
            originals = set(observations)

            # Execute tasks with dependency management
            futures = []
            retry_after = 0.25  # Retry interval for dependent tasks

            with ThreadPoolExecutor() as executor:
                for task in self.tasks:
                    deps = task["dependencies"]
                    task_names[task["idx"]] = (
                        task["tool"]
                        if isinstance(task["tool"], str)
                        else task["tool"].name
                    )
                    args_for_tasks[task["idx"]] = task["args"]

                    if deps and any(dep not in observations for dep in deps):
                        # Schedule dependent task for later execution
                        futures.append(
                            executor.submit(
                                self.schedule_pending_task,
                                task,
                                observations,
                                retry_after,
                            )
                        )
                    else:
                        # Execute independent task immediately
                        self.schedule_task(
                            dict(task=task, observations=observations), config
                        )

                # Wait for all tasks to complete
                wait(futures)

            # Format results as tool messages
            return self._format_tool_messages(
                observations, originals, task_names, args_for_tasks
            )

        except Exception as e:
            logger.error("ERROR in scheduler: %s", str(e))
            raise

    def _format_tool_messages(
        self,
        observations: Dict[int, Any],
        originals: Set[int],
        task_names: Dict[int, str],
        args_for_tasks: Dict[int, Any],
    ) -> List[FunctionMessage]:
        # Convert observations to new tool messages to add to the state
        new_observations = {
            k: (task_names[k], args_for_tasks[k], observations[k])
            for k in sorted(observations.keys() - originals)
        }

        tool_messages = []
        for k, (name, task_args, obs) in new_observations.items():
            tool_message = FunctionMessage(
                name=name,
                content=(
                    str(obs)
                    if not isinstance(obs, dict) or "message" not in obs
                    else obs["message"]
                ),
                response_metadata=(
                    {}
                    if not isinstance(obs, dict) or "metadata" not in obs
                    else obs["metadata"]
                ),
                additional_kwargs={"idx": k, "args": task_args},
                tool_call_id=k,
            )
            tool_messages.append(tool_message)

        return tool_messages


if __name__ == "__main__":
    scheduler = Scheduler()
    print(scheduler._resolve_task_args("[ $1, $2  ]", {1: "hello", 2: "world"}))
