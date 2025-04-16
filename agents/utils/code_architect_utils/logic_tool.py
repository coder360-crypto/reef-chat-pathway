# Import required libraries
import os
import sys

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and OpenAI chat model
from config import AgentsConfig
from langchain_openai import ChatOpenAI


def logic_tool(
    input_string: str,
    request: str,
    output_type: str = "str",
    return_explanation: bool = False,
    llm: str = "gpt-4o",
):
    """
    A tool that processes logical operations on input strings based on specific requests.
    
    Args:
        input_string (str): The input text to be processed
        request (str): The specific operation or query to be performed on the input
        output_type (str): Expected type of the output (default: "str")
        return_explanation (bool): Whether to return explanation along with response (default: False)
        llm (str): The language model to use (default: "gpt-4o")
    
    Returns:
        Union[str, Tuple[str, str]]: Either the response string alone, or a tuple of (response, explanation)
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(model=llm, api_key=AgentsConfig().OPENAI_API_KEY)

    # Make API call to get the response
    response = llm.invoke(
        input=[
            {
                "role": "system",
                "content": """
            You are a logic tool. You will be given an input string, a request and an output type.
            Your job is to use the input string to answer the request.
            Enclose your response between <response> and </response> tags.
            Also enclose an explanation between <explanation> and </explanation> tags.
            Ensure that the explanation is consistent with the response.
            Strictly follow the request and the output type.
             
            Example:
            <input>The capital expenditures for company X in 2018 is 100 million dollars.</input>
            <request>Extract the capital expenditures for company X in 2018.</request>
            <output_type>int</output_type>
            <response>100000000</response>
            <explanation>The capital expenditures for company X in 2018 is 100 million dollars.</explanation>
            """,
            },
            {
                "role": "user",
                "content": f"""
            <input>{input_string}</input>
            <request>{request}</request>
            <output_type>{output_type}</output_type>
            """,
            },
        ]
    )

    # Extract and return response (and explanation if requested)
    if return_explanation:
        return (
            response.content[
                response.content.find("<response>")
                + len("<response>") : response.content.find("</response>")
            ],
            response.content[
                response.content.find("<explanation>")
                + len("<explanation>") : response.content.find("</explanation>")
            ],
        )
    else:
        return response.content[
            response.content.find("<response>")
            + len("<response>") : response.content.find("</response>")
        ]
