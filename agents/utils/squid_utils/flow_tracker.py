import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# Track tool usage with timestamps
@dataclass
class ToolUsage:
    """
    Tracks individual tool usage instances with timestamps.
    
    Attributes:
        tool_name (str): Name of the tool being used
        input_args (dict): Arguments passed to the tool
        output (str): Output returned by the tool
        timestamp (str): ISO format timestamp of tool usage
    """
    tool_name: str
    input_args: dict
    output: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Track individual specialist's task details and progress
@dataclass
class SpecialistTask:
    """
    Tracks the details and progress of an individual specialist's assigned task.
    
    Attributes:
        specialist_name (str): Name of the specialist
        assigned_task (str): Raw task assignment text
        task_description (Optional[str]): Description of the task
        key_entities (List[str]): List of entities to analyze
        key_tasks (List[str]): List of specific tasks to perform
        tool_usage (List[ToolUsage]): Record of tools used
        final_response (Optional[str]): Specialist's final output
        plottable_information (Optional[str]): Data that can be visualized
        iteration_mode (int): Number of iterations required
        current_iteration (int): Current iteration number
        intermediate_results (Optional[dict]): Results from intermediate steps
        start_time (str): ISO format task start timestamp
        end_time (Optional[str]): ISO format task completion timestamp
    """
    specialist_name: str
    assigned_task: str
    task_description: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    key_tasks: List[str] = field(default_factory=list)
    tool_usage: List[ToolUsage] = field(default_factory=list)
    final_response: Optional[str] = None
    plottable_information: Optional[str] = None
    iteration_mode: int = field(default=1)
    current_iteration: int = field(default=1)
    intermediate_results: Optional[dict] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None


# Main class to track the entire task workflow
@dataclass
class TaskFlow:
    """
    Main class for tracking the entire workflow of a task, including specialist assignments and final compilation.
    
    Attributes:
        original_query (str): The initial user query
        specialist_tasks (Dict[str, SpecialistTask]): Mapping of specialist names to their tasks
        final_compilation (Optional[str]): Final compiled response
        start_time (str): ISO format workflow start timestamp
        end_time (Optional[str]): ISO format workflow completion timestamp
    """
    original_query: str
    specialist_tasks: Dict[str, SpecialistTask] = field(default_factory=dict)
    final_compilation: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    def add_specialist_task(
        self, specialist_name: str, task: str, iteration_mode: int = 1
    ):
        """Record new specialist task assignment with structured components"""
        # Parse the formatted task string into components
        task_parts = task.split("\n")

        # Extract task description
        task_description = next(
            (
                line.replace("Main Task:", "").strip()
                for line in task_parts
                if line.strip().startswith("Main Task:")
            ),
            "",
        )

        # Extract key entities
        entities_start = (
            task_parts.index("Key Entities to Analyze:")
            if "Key Entities to Analyze:" in task_parts
            else -1
        )
        tasks_start = (
            task_parts.index("Specific Tasks to Perform:")
            if "Specific Tasks to Perform:" in task_parts
            else len(task_parts)
        )

        key_entities = [
            line.replace("- ", "").strip()
            for line in task_parts[entities_start + 1 : tasks_start]
            if line.strip().startswith("- ")
        ]

        # Extract key tasks
        focus_start = (
            task_parts.index("Focus Areas:")
            if "Focus Areas:" in task_parts
            else len(task_parts)
        )
        key_tasks = [
            line.replace("- ", "").strip()
            for line in task_parts[tasks_start + 1 : focus_start]
            if line.strip().startswith("- ")
        ]

        # Create the specialist task with all components and iteration mode
        self.specialist_tasks[specialist_name] = SpecialistTask(
            specialist_name=specialist_name,
            assigned_task=task,
            task_description=task_description,
            key_entities=key_entities,
            key_tasks=key_tasks,
            iteration_mode=iteration_mode,
            current_iteration=1,
        )

    def add_tool_usage(
        self, specialist_name: str, tool_name: str, input_args: dict, output: str
    ):
        """Record tool usage by a specialist"""
        if specialist_name in self.specialist_tasks:
            self.specialist_tasks[specialist_name].tool_usage.append(
                ToolUsage(tool_name=tool_name, input_args=input_args, output=output)
            )

    def set_specialist_response(
        self, specialist_name: str, response: str, plottable_info: str = None
    ):
        """Record specialist's final response with plottable information"""
        if specialist_name in self.specialist_tasks:
            self.specialist_tasks[specialist_name].final_response = response
            self.specialist_tasks[specialist_name].plottable_information = (
                plottable_info
            )
            self.specialist_tasks[specialist_name].end_time = datetime.now().isoformat()

    def set_final_compilation(self, compilation: str):
        """Record final compiled response"""
        self.final_compilation = compilation
        self.end_time = datetime.now().isoformat()

    def set_image_path(self, imgpath: str = None):
        self.img_path = imgpath

    def to_dict(self) -> dict:
        """Convert the task flow to a dictionary with iteration tracking"""
        return {
            "original_query": self.original_query,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "specialist_tasks": {
                name: {
                    "specialist_name": task.specialist_name,
                    "assigned_task": task.assigned_task,
                    "task_description": task.task_description,
                    "key_entities": task.key_entities,
                    "key_tasks": task.key_tasks,
                    "tool_usage": [vars(usage) for usage in task.tool_usage],
                    "final_response": task.final_response,
                    "plottable_information": task.plottable_information,
                    "iteration_mode": task.iteration_mode,
                    "current_iteration": task.current_iteration,
                    "intermediate_results": task.intermediate_results,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                }
                for name, task in self.specialist_tasks.items()
            },
            "final_compilation": self.final_compilation,
            "plot_path": getattr(self, "img_path", None),  # Safely handle img_path
        }

    def save_to_file(self, filename: str):
        """Save the task flow to a JSON file"""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
