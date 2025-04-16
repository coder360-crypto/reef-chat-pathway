# Import required system modules
import os
import sys

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and OpenAI chat model
from config import AgentsConfig
from langchain_openai import ChatOpenAI


def final_formatter(output: str, query: str) -> str:
    """
    Format the output according to the query using GPT-4 to ensure logical and semantic correctness.

    Args:
        output (str): The raw output to be formatted
        query (str): The original user query

    Returns:
        str: Formatted response without metadata tags
    """
    # Initialize ChatOpenAI instance with GPT-4
    llm = ChatOpenAI(model="gpt-4o", api_key=AgentsConfig().OPENAI_API_KEY)

    # Make API call with system and user messages
    response = llm.invoke(
        input=[
            {
                "role": "system",
                "content": """
                    You are a final formatting agent. You will be given an output and a query.
                    Your job is to format the output according to the query and make it logically and semantically correct.
                    Do not add any additional information to the output.
                    Ensure that you do not use words like "input" and "output". Write the answer as if you were the one answering the query.
                    Enclose your response between <response> and </response> tags.
                """,
            },
            {
                "role": "user",
                "content": f"<output>{output}</output><query>{query}</query>",
            },
        ]
    )

    # Extract content between response tags
    return response.content[
        response.content.find("<response>")
        + len("<response>") : response.content.find("</response>")
    ]
