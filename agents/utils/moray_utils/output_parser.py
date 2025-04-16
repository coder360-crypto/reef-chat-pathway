"""
Output parser for the MORAY system.

This module handles:
- Parsing LLM outputs into executable tasks
- Tool resolution and validation
- Dependency tracking
- Error handling and reporting
"""

import ast
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Regular expression patterns
THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = "<END_OF_PLAN>"


class Task(TypedDict):
    """
    Represents a single task in the execution plan.

    Attributes:
        idx: Task index for ordering and dependencies
        tool: Tool to execute
        args: Arguments for the tool
        dependencies: List of dependent task indices
        thought: Optional reasoning about the task
    """

    idx: int
    tool: BaseTool
    args: list
    dependencies: Dict[str, list]
    thought: Optional[str]


def _ast_parse(arg: str) -> Any:
    """
    Safely parse argument strings into Python objects.

    Args:
        arg: String to parse

    Returns:
        Parsed Python object or original string if parsing fails
    """
    try:
        return ast.literal_eval(arg)
    except:  # noqa
        return arg


def _parse_moray_action_args(args: str, tool: Union[str, BaseTool]) -> list[Any]:
    """
    Parse tool arguments from a string representation.

    Args:
        args: String containing tool arguments
        tool: Tool that will use the arguments

    Returns:
        List of parsed arguments
    """
    if args == "" or isinstance(tool, str):
        return ()

    extracted_args = {}
    tool_key = None
    prev_idx = None

    # Parse each argument based on tool's expected parameters
    for key in tool.args.keys():
        if f"{key}=" in args:
            idx = args.index(f"{key}=")

            if prev_idx is not None:
                extracted_args[tool_key] = _ast_parse(
                    args[prev_idx:idx].strip().rstrip(",")
                )

            args = args.split(f"{key}=", 1)[1]
            tool_key = key
            prev_idx = 0

    # Handle the last argument
    if prev_idx is not None:
        extracted_args[tool_key] = _ast_parse(
            args[prev_idx:].strip().rstrip(",").rstrip(")")
        )

    return extracted_args


def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Dict[str, Any]
) -> dict[str, list[str]]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        return list(range(1, idx))
    return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]


def get_tool_mapping(tools: Sequence[BaseTool]) -> Dict[str, str]:
    """
    Create mappings between tool names and their variations.

    Args:
        tools: Available tools

    Returns:
        Dictionary mapping tool name variations to canonical names
    """
    # Define common variations for known tools
    tool_variations = {
        "yahoo_finance_news": [
            "finance",
            "yahoo_finance",
            "yahoo",
            "stock",
            "stocks",
            "financial_news",
            "market_news",
        ],
        "wikipedia": [
            "wiki",
            "wikipedia_search",
            "wikipedia_search_results_json",
            "wiki_search",
            "encyclopedia",
        ],
        "tavily_search_results_json": [
            "search",
            "tavily",
            "web_search",
            "internet_search",
            "google",
        ],
        "math": [
            "calculate",
            "calculator",
            "computation",
            "compute",
            "mathematical",
            "arithmetic",
        ],
    }

    # Build complete mapping
    mapping = {}
    available_tools = {tool.name.lower(): tool.name for tool in tools}

    for tool_name in available_tools:
        # Add exact name mapping
        mapping[tool_name] = available_tools[tool_name]

        # Add variations if they exist
        if tool_name in tool_variations:
            for variation in tool_variations[tool_name]:
                mapping[variation.lower()] = available_tools[tool_name]

    return mapping


def instantiate_task(
    tools: Sequence[BaseTool],
    idx: int,
    tool_name: str,
    args: Union[str, Any],
    thought: Optional[str] = None,
) -> Task:
    """
    Create a task instance with proper tool resolution.

    Args:
        tools: Available tools
        idx: Task index
        tool_name: Name of tool to use
        args: Tool arguments
        thought: Optional reasoning about the task

    Returns:
        Instantiated task

    Raises:
        OutputParserException: If tool cannot be found
    """
    # Handle join tool specially
    if tool_name == "join":
        return Task(
            idx=idx,
            tool="join",
            args=args,
            dependencies=list(range(1, idx)),
            thought=thought,
        )

    # Get tool mappings and normalize name
    tool_mapping = get_tool_mapping(tools)
    normalized_tool_name = tool_mapping.get(tool_name.lower(), tool_name)

    try:
        # Find matching tool
        tool = next(
            (t for t in tools if t.name.lower() == normalized_tool_name.lower()), None
        )

        if tool is None:
            raise OutputParserException(
                _generate_tool_suggestion_message(tool_name, tools, tool_mapping)
            )

        # Parse arguments and get dependencies
        tool_args = _parse_moray_action_args(args, tool)
        dependencies = _get_dependencies_from_graph(
            idx, normalized_tool_name, tool_args
        )

        return Task(
            idx=idx,
            tool=tool,
            args=tool_args,
            dependencies=dependencies,
            thought=thought,
        )

    except Exception as e:
        logger.info(
            f"Error instantiating tool '{tool_name}': {str(e)}. Defaulting to join."
        )
        return Task(
            idx=idx,
            tool="join",
            args=args,
            dependencies=list(range(1, idx)),
            thought=thought,
        )


def _generate_tool_suggestion_message(
    tool_name: str, tools: Sequence[BaseTool], variations: Dict[str, str]
) -> str:
    """Generate helpful error message with tool suggestions."""
    available_tools = [t.name for t in tools]
    msg = [
        f"Tool '{tool_name}' not found. Available tools: {available_tools}\n",
        "You can also use these variations:\n",
    ]

    # Group variations by actual tool name
    for actual_tool in available_tools:
        aliases = [k for k, v in variations.items() if v == actual_tool]
        if aliases:
            msg.append(f"- {actual_tool}: {', '.join(aliases)}\n")

    return "".join(msg)


class MORAYPlanParser(BaseTransformOutputParser[dict]):
    """
    Parser for LLM outputs into executable tasks.

    This parser handles:
    - Streaming output processing
    - Task instantiation
    - Thought extraction
    - Error recovery
    """

    tools: List[BaseTool]

    def parse(self, text: str) -> List[Task]:
        """Parse complete text into tasks."""
        logger.info(f"Parsing input: {text}")
        return list(self._transform([text]))

    def stream(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Task]:
        """Stream parse results."""
        yield from self.transform([input], config, **kwargs)

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        """Transform input stream into tasks."""
        texts = []
        thought = None

        for chunk in input:
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self.ingest_token(text, texts, thought):
                yield task

        if texts:
            task, _ = self._parse_task("".join(texts), thought)
            if task:
                yield task

    def ingest_token(
        self, token: str, buffer: List[str], thought: Optional[str]
    ) -> Iterator[Tuple[Optional[Task], str]]:
        """Process incoming tokens and yield tasks."""
        buffer.append(token)
        if "\n" in token:
            buffer_ = "".join(buffer).split("\n")
            suffix = buffer_[-1]

            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought

            buffer.clear()
            buffer.append(suffix)

    def _parse_task(
        self, line: str, thought: Optional[str] = None
    ) -> Tuple[Optional[Task], Optional[str]]:
        """
        Parse a single line into a task.

        Args:
            line: Text line to parse
            thought: Current thought context

        Returns:
            Tuple of (Task if found, updated thought)
        """
        task = None

        if match := re.match(THOUGHT_PATTERN, line):
            thought = match.group(1)
        elif match := re.match(ACTION_PATTERN, line):
            idx, tool_name, args, _ = match.groups()
            idx = int(idx)
            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )
            thought = None

        return task, thought
