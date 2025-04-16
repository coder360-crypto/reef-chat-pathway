import logging
import os
import sys
from typing import List, Union

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import asyncio

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FinalResponse(BaseModel):
    """Thought about the final response."""

    thought: Union[str, dict] = Field(description="Thought about the final response.")


class Replan(BaseModel):
    """Thought about the replan."""

    feedback: Union[str, dict] = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed or what is missing or ambiguous."
    )


class JoinOutputs(BaseModel):
    """Generate the complete and correct final response and decide whether to replan or not."""

    response: str = Field(
        description="A complete and correct final response along with all the relevant and correct information gathered so far in a properly structured markdown format for ease of rendering and readability."
    )
    action: Union[FinalResponse, Replan]


class Joiner:
    """
    Synthesizes and formats final responses from tool execution results.

    This class handles:
    - Combining results from multiple tools into a coherent response
    - Deciding whether to replan based on result analysis
    - Formatting responses in markdown for readability
    - Managing response structure and quality

    Attributes:
        runnable: Chain that processes messages and generates structured output
    """

    def __init__(self, llm, query):
        """
        Initialize the Joiner with a language model.

        Args:
            llm: Language model for response synthesis
        """
        joiner_prompt = ChatPromptTemplate.from_messages(
            [   
                SystemMessage(
                    content="""
Generate a response to the user query as an expert human analyst that generates a complete and correct response to the query through chain-of-thought reasoning.
Follow this Chain of Thought process for your internal analysis:

Begin by carefully analyzing the query to understand its required depth of analysis. For simple queries like asking about Tesla's current stock price or India's prime minister, provide concise responses with minimal markdown formatting. For complex queries, identify core information requirements, implicit analytical needs, specific metrics or calculations needed, break down the query components, and identify key analytical dimensions required.
Next, thoroughly examine all provided data by identifying relevant numerical data points, clauses, and other relevant information, recognizing relationships between different metrics, entities and clauses, and spotting patterns and anomalies. When dealing with financial or numerical data, verify calculation methodologies, cross-validate related metrics, consider contextual significance, and analyze year-over-year or period-over-period changes. When dealing with legal data, analyze the information for specific clauses or sentences that user might have been referring to.
For the analytical reasoning phase, examine each key finding by explaining the logical path to the conclusion, supporting with specific data points, considering implications and relationships, and validating against other metrics. When performing calculations, show step-by-step computation, explain methodology choices, verify accuracy and reasonableness, and consider alternative approaches.
Finally, formulate the response by structuring it based on analytical flow - present key findings in order of significance, support each point with specific data or information, and link related insights. For numerical or tabular data, use proper markdown tables, maintain exact precision, include relevant headers and labels, and add explanatory notes where needed. The response must be complete, properly structured, and focus only on verified information.
After the response is complete, decide whether to provide a final thought or feedback for replanning based on the thorough analysis.

DECISION LOGIC:
   Choose ONE based on thorough analysis:
   
   A. FinalResponse:
      - thought: Detailed reasoning showing:
        * Why the analysis is complete
        * How all requirements are met
        * Validation of calculations
        * Confidence in conclusions
      
   B. Replan:
      - feedback: Systematic analysis of:
        * Information gaps identified
        * Calculation limitations
        * Areas needing clarification
        * Specific additional data needed
"""
                ),
                HumanMessage(
                    content=f"Query: {query}"
                ),
                MessagesPlaceholder(variable_name="messages"),
                SystemMessage(
                    content="""
Internal analytical process:

1. COMPREHENSION PHASE:
   - Analyze query requirements thoroughly
   - Identify all relevant data points
   - Map relationships between information
   - Note any potential gaps

2. SYNTHESIS PHASE:
   - Integrate information systematically
   - Validate all calculations
   - Cross-reference related data
   - Build logical connections

3. VALIDATION PHASE:
   - Verify completeness of analysis
   - Check calculation accuracy
   - Confirm logical consistency
   - Assess need for additional data

4. DECISION PHASE:
   Either:
   - Provide final response with complete reasoning
   - Or identify specific additional information needed

Review the execution results and provide:
    1. A complete response with all relevant information gathered so far based on the complexity of the query
    2. Either:
    - Final thought if information is complete
    - Feedback for replanning if more information is needed or the previous attempt information is insufficient or ambiguous

Remember:
- Response must be correct and complete regardless of decision
- Markdown formatting is required if needed
- All current relevant and correct information must be included
- Decision must be clearly justified
- Tables must be properly formatted and preserved
"""
                ),
                SystemMessage(
                    content="""
RESPONSE STRUCTURE:

    - Provide the response based on the complexity of the query. Don't provide over explanation for simple queries.
    - Use clear and formal markdown formatting
    - Include ONLY verified and relevant information gathered from all the tools based on the query
    - IMPORTANT: If tools return any tabular data, ALWAYS include it in markdown table format
    - For numerical tables:
        * Preserve all numbers exactly as returned
        * Maintain column headers and row labels
        * Use proper markdown table alignment (:|--:| for center, :--|-- for left, --:|-- for right)
    - Attribute sources when available
    - Must be complete and readable regardless of decision to replan
    - Must be properly structured with headers and sections if needed based on the complexity of the query
    - Focus on WHAT IS KNOWN, not what isn't
    - NEVER include phrases like:
        ✗ "I was unable to"
        ✗ "I apologize"
        ✗ "Unfortunately"
        ✗ "I couldn't"
        ✗ "due to technical issues"
        ✗ Any mentions of errors or failures

EXAMPLE RESPONSE:
{
    "response": "Correct, complete and relevant response compiled from all the information gathered so far in a properly structured markdown format for ease of rendering and readability. Remember not to include any personal negetive feedback or information that is not gathered, not relevant, or not complete.",
    "action": ONE OF:
        {
            "thought": "Reasoning for completion"
        }
        OR
        {
            "feedback": "Detailed replan analysis to get more or complete information or to fix the previous mistakes"
        }
}

TABLE FORMATTING RULES:
- Always preserve exact numerical values
- Use proper markdown table syntax:  ```
  | Header 1 | Header 2 | Header 3 |
  |:---------|:--------:|----------:|
  | Left     | Center   | Right    |  ```
- For financial data:
  * Right-align numerical columns
  * Include currency symbols if present
  * Maintain decimal precision
- Include table captions or titles when available
- Preserve any footnotes or annotations
- If any tool returns tabular data, it MUST be included in the response
- Use proper markdown table formatting
- Preserve numerical precision and alignment
- Include table context and descriptions
                          
CRITICAL ANALYTICAL RULES:

✓ Use professional, authoritative tone
✓ Clear and formal Markdown formatting is required if needed
✓ Must be properly structured with headers and sections if needed based on the complexity of the query
✓ Show reasoning process
✓ Cross-reference related metrics
✓ Maintain numerical precision
✓ Support conclusions with data
✓ Consider multiple analytical angles
✓ Explain methodology choices
✗ No unsupported conclusions
✗ No speculation beyond data
✗ No missing step explanations
✗ No unverified assumptions

NOTE for IPR Agent:
- Return the final output of this agent without any changes.
- Do not summarize the output of this agent.
- Do not add or remove any part of this output.
- Retain the structure of the output and all the links, serial number, etc. provided by the agent in the final output.
"""
                )
            ]
        )
        if 'gpt' not in llm.model_name:
            joiner_prompt = joiner_prompt[:2000]
        self.runnable = joiner_prompt | llm.with_structured_output(JoinOutputs)

    @staticmethod
    def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
        """
        Parse the joiner output into messages.

        Args:
            decision: Structured output from joiner

        Returns:
            List of messages containing response and decision
        """
        # Create response message
        response = AIMessage(
            content=(
                decision.response["message"]
                if "message" in decision.response
                else str(decision.response)
            ),
            response_metadata=(
                decision.response["metadata"] if "metadata" in decision.response else {}
            ),
        )

        # Handle replan vs final response
        if isinstance(decision.action, Replan):
            return {
                "messages": [
                    SystemMessage(content=f"Feedback: {decision.action.feedback}"),
                    response,
                ]
            }
        thought = (
            decision.action.thought["message"]
            if isinstance(decision.action.thought, dict)
            else decision.action.thought
        )
        return {"messages": [AIMessage(content=f"Thought: {thought}"), response]}

    @staticmethod
    def select_recent_messages(state) -> dict:
        """
        Select the most recent set of messages up to the last human message.

        Args:
            state: Current message state

        Returns:
            Dict containing selected messages
        """

        # Collect messages until we find a human message
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        # Optimize message selection by using list comprehension
        human_msg_idx = next(
            (
                i
                for i, msg in enumerate(reversed(messages))
                if isinstance(msg, HumanMessage)
            ),
            len(messages),
        )
        # Add 1 to include messages after human message but exclude system context message
        human_msg_idx = human_msg_idx + 1
        selected = messages[-human_msg_idx:]

        return {"messages": selected}

    def get_chain(self):
        """Get the processing chain for joining responses."""
        return (
            Joiner.select_recent_messages | self.runnable | Joiner._parse_joiner_output
        )


if __name__ == "__main__":
    import os
    import time

    from config import AgentsConfig as Config
    from langchain_openai import ChatOpenAI
    from tqdm import tqdm

    print("Testing Joiner...")

    # Initialize components
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=Config().OPENAI_API_KEY)

    joiner = Joiner(llm, query="What is the balance sheet and income statement of Tesla?")

    # Extract 5 test messages
    test_messages = [
        {
            "messages": [
                HumanMessage(
                    content="What is the balance sheet and income statement of Tesla?"
                ),
                AIMessage(
                    content="""
                 The balance sheet of Tesla is as following:
                 ```
                 | Assets | Liabilities |
                 |--------|-------------|
                 | 100    | 50          |
                 ```

                 The income statement of Tesla is as following:
                 ```
                 | Revenue | Expenses |
                 |--------|-------------|
                 | 100    | 50          |
                 ```
                 """
                ),
            ]
        }
    ]

    # Test latency
    total_time = 0

    for state in tqdm(test_messages, desc="Running tests"):
        start_time = time.time()

        # Process using the joiner chain
        chain = joiner.get_chain()
        outputs = []
        for result in chain.stream(state):
            outputs.append(result)

        elapsed = time.time() - start_time
        total_time += elapsed

        print(f"\nTest completion time: {elapsed:.2f} seconds")
        # print(f"Response preview: {str(outputs[-1]['messages'][-1].content)[:150]}...")

    avg_time = total_time / (len(test_messages))
    print(f"\nAverage processing time: {avg_time:.2f} seconds per message")
