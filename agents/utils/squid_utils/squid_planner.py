import logging
import os
import sys
from typing import Any, Dict, List

from config import AgentsConfig as Config

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from utils.squid_utils.squid_prompts import TOOL_CAPABILITIES

config = Config()


class PlanningOutput(BaseModel):
    """The final structured output after thought interpretation.
    
    Args:
        selected_specialists (List[str]): List of selected specialists
        query_type (str): Type of analysis needed
        complexity (str): Complexity level (simple, moderate, complex)
        response_length (str): Expected response length (short, medium, long)
        tasks (Dict[str, Dict[str, Any]]): Tasks for each specialist
    """

    selected_specialists: List[str] = Field(description="List of selected specialists")
    query_type: str = Field(description="Type of analysis needed")
    complexity: str = Field(description="Complexity level (simple, moderate, complex)")
    response_length: str = Field(
        description="Expected response length (short, medium, long)"
    )
    tasks: Dict[str, Dict[str, Any]] = Field(description="Tasks for each specialist")


class ThoughtBasedPlanner:
    """Handles the two-stage planning process: thinking and interpretation.
    
    Args:
        llm: Language model for thinking stage
        prompt_store: Store containing prompts for planning
    """

    def __init__(self, llm, prompt_store):
        # Initialize main LLM for thinking stage
        self.llm = llm
        self.prompt_store = prompt_store
        
        # Setup GPT-4 interpreter with structured output capability
        self.interpreter_llm = ChatOpenAI(
            model="gpt-4o", temperature=0, api_key=config.OPENAI_API_KEY
        )
        self.interpreter_llm = self.interpreter_llm.with_structured_output(
            PlanningOutput, method="json_mode"
        )

    def _generate_thinking_prompt(self, query: str) -> str:
        """Generates the prompt for the thinking stage.
        
        Args:
            query (str): User's input query
            
        Returns:
            str: Generated thinking prompt
        """
        return self.prompt_store.return_plan_thinking_prompt(query, TOOL_CAPABILITIES)

    def _generate_interpreter_prompt(self, thoughts: str, query: str = None) -> str:
        """Generates the prompt for the interpretation stage.
        
        Args:
            thoughts (str): Generated thoughts from thinking stage
            query (str, optional): Original user query
            
        Returns:
            str: Generated interpreter prompt
        """
        return self.prompt_store.return_thought_structuring_prompt(query, thoughts)

    def _format_tools(self, tools: Dict[str, List[str]]) -> str:
        """Formats the available tools for the prompt.
        
        Args:
            tools (Dict[str, List[str]]): Dictionary mapping specialists to their tools
            
        Returns:
            str: Formatted string of tools
        """
        formatted = []
        for specialist, tool_list in tools.items():
            tools_str = ", ".join(tool_list)
            formatted.append(f"{specialist}:\n  Tools: {tools_str}")
        return "\n".join(formatted)

    async def generate_plan(self, query: str) -> PlanningOutput:
        """Main method to generate a plan through thinking and interpretation.
        
        Args:
            query (str): User's input query
            
        Returns:
            PlanningOutput: Structured plan containing specialists, tasks, and metadata
            
        Raises:
            ValueError: If no tasks are generated or if tasks are missing for specialists
            ValidationError: If structured output validation fails
            Exception: For other unexpected errors
        """
        try:
            # Stage 1: Generate free-form thoughts about the query
            thinking_prompt = self._generate_thinking_prompt(query)
            thoughts_response = await self.llm.ainvoke(
                [
                    SystemMessage(content=thinking_prompt),
                    HumanMessage(content="Here is the query: " + query),
                ],
            )
            print("#" * 50)
            print("Thought process completed. Thoughts:", thoughts_response.content)

            # Stage 2: Convert thoughts into structured plan
            interpreter_prompt = self._generate_interpreter_prompt(
                thoughts_response.content
            )
            structured_plan = None
            print("Passing to the structured planner:")

            # First attempt at structured planning
            try:
                structured_plan = await self.interpreter_llm.ainvoke(
                    [
                        SystemMessage(content=interpreter_prompt),
                        HumanMessage(
                            content=(
                                "ENSURE THAT YOU RESPONSE IN THE SPECIFIED STRUCTURED OUTPUT FORMAT CORRECTLY FOR GIVEN QUERY AND THOUGHT PROCESS"
                                + "Here is the query: "
                                + query
                                + "\n\n Here are the thoughts: "
                                + thoughts_response.content
                            )
                        ),
                    ],
                )
            except ValidationError as e:
                # Retry once if first attempt fails validation
                print("Error in getting structured output:", e)
                print("[Retrying again.....]")
                structured_plan = await self.interpreter_llm.ainvoke(
                    [
                        SystemMessage(content=interpreter_prompt),
                        HumanMessage(
                            content="Here is the query: "
                            + query
                            + "\n\n Here are the thoughts: "
                            + thoughts_response.content
                            + "ENSURE RETURNING RESULT IN CORRECT FORMAT"
                        ),
                    ],
                )

            print("#" * 50)
            print("Structured plan generated:", structured_plan)

            # Validate plan has required tasks
            if not hasattr(structured_plan, "tasks") or not structured_plan.tasks:
                raise ValueError("No tasks generated in structured plan")

            # Ensure all selected specialists have assigned tasks
            if not all(
                spec in structured_plan.tasks
                for spec in structured_plan.selected_specialists
            ):
                missing = [
                    spec
                    for spec in structured_plan.selected_specialists
                    if spec not in structured_plan.tasks
                ]
                raise ValueError(f"Missing tasks for specialists: {missing}")
            return structured_plan

        except Exception as e:
            # Log error details for debugging
            print(f"Error in generating plan: {str(e)}")
            print("Thought process output:", thoughts_response.content)
            print(
                "Structured plan attempt:",
                structured_plan if "structured_plan" in locals() else "Not generated",
            )
            raise e
