import json
import logging

from config import AgentsConfig as Config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

config = Config()

# Use __name__ to get the correct module name
logger = logging.getLogger(__name__)


async def analyze_query(
    query: str, agent: str, llm: str = "llama-3.1-70b-versatile"
) -> dict:
    """Analyzes query and determines handling strategy with a single LLM call"""
    try:
        if "gpt" not in llm:
            analysis_llm = ChatGroq(
                model=llm, temperature=0, api_key=config.GROQ_API_KEY
            )
        else:
            analysis_llm = ChatOpenAI(
                model=llm, temperature=0, api_key=config.OPENAI_API_KEY
            )

        combined_prompt = f"""
        You are a legal and financial assistant. Your job is to analyze the user's query and determine the appropriate handling.
        Your name is "Reef".
        
        Analyze the following query generate the thoughts about the query and what all information is required to answer the query. Based on the the thoughts determine the appropriate handling:
        Query: "{query}"

        Follow these steps in order:

        1. First check if this is a CONVERSATIONAL query:
           - Basic greetings or casual chat ("how are you", "hello")
           - Basic math calculations
           - Expressing emotions or opinions
           - Any personal information from the user not requiring any analysis
           - Any questions about the user themselves or their preferences
           - Any query asking about you or your capabilities
           Examples: "How are you?", "What's 15% of 200?", "Why is the market so frustrating?"

        2. Then check if the query is unclear:
            - If the query is unclear, ask the user to clarify their query. You may recieve some information, but if unclear make sure to say so.
            - If the key entities in the query are not referenced clearly, ask the user to clarify which entities they are referring to.
            Examples: 
                <provided_query>
                    The revenue of Company X is .....
                    The revenue of Company Y is .....
                    What is the revenue of this Company?
                </provided_query>
                <clarification_response>
                    Could you please clarify which company you're referring to?
                </clarification_response>
        3. If not conversational, handle based on the below rules:
           
           Choose between MORAY or SQUID based on:
           SQUID for:
           - Very complex and multi-layered queries.
           - Very long, detaild analysis and response queries

           MORAY for:
           - All other queries except conversational
           Examples: "Review this contract", "Explain new regulations", "What is the revenue of X?"

        Respond ONLY with a JSON object in this EXACT format (no other text):
        {{
            "agent": "conversational" or "clarification_response" or "moray" or "squid",
            "response": "<<insert response from Reef>>"
        }}
    """

        response = analysis_llm.invoke(combined_prompt)

        # Clean the response content
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1]
        elif content.startswith("```"):
            content = content.split("```")[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        result = json.loads(content)

        # For conversational queries, return with response
        if result["agent"] == "conversational":
            return {"agent": "conversational", "response": result["response"]}

        # For specific agents (moray/squid), return that agent
        if agent == "auto":
            return {"agent": result["agent"]}
        else:
            return {"agent": agent}

    except Exception as e:
        logger.error(f"Error in query analysis: {str(e)}")
        return {
            "agent": "moray",
        }


def evaluate_response(result, agent_type, llm: str = "llama-3.1-70b-versatile"):
    """
    Evaluate the response to determine if it contains errors using LLM checker.
    Returns (is_valid, error_message)
    """
    if "gpt" not in llm:
        llm = ChatGroq(model=llm, temperature=0.1, api_key=config.GROQ_API_KEY)
    else:
        llm = ChatOpenAI(model=llm, temperature=0.1, api_key=config.OPENAI_API_KEY)

    error_check_prompt_template = """
    Analyze the following text to determine if it contains any indication of failure, error, or inability to complete the task.

    Instructions:
    - If the text contains ANY of these patterns, respond with 'false: <one line reason>' where the reason explains the error:
      1. Explicit error messages:
         - "Error: [description]"
         - "Failed to [action]"
         - "Unable to complete [task]"
      
      2. Tool/System failures:
         - "Tool failed to respond"
         - "Failed to run [tool/analysis]"
         - "Could not process"
         
      3. Incomplete or failed responses:
         - "I cannot provide"
         - "Failed to generate response"
         - "Analysis was incomplete"

    - If no errors are found, respond with just 'true'
    - Do not respond with 'false' if:
      - The text discusses errors or failures as topics
      - The word 'fail' or 'error' is used in a different context
      - The response is complete and valid but mentions past failures

    Text to analyze: {text_to_analyze}

    Response (true or false: reason):"""

    if agent_type == "moray":
        # Extract final content from moray result
        final_content = []

        # Check if results exist in the response
        if "results" not in result:
            return False, "No results found in moray response"

        for step in result["results"]:
            # Check for join step which contains the final response
            if "join" in step and step["join"].get("messages"):
                for msg in step["join"]["messages"]:
                    if isinstance(msg, dict):
                        if "content" in msg:
                            final_content.append(msg["content"])
                    # Handle AIMessage objects
                    elif hasattr(msg, "content"):
                        final_content.append(msg.content)

        # If no content was found in join messages, try plan_and_schedule
        if not final_content:
            for step in result["results"]:
                if "plan_and_schedule" in step and step["plan_and_schedule"].get(
                    "messages"
                ):
                    for msg in step["plan_and_schedule"]["messages"]:
                        if isinstance(msg, dict):
                            if "content" in msg:
                                final_content.append(msg["content"])
                        elif hasattr(msg, "content"):
                            final_content.append(msg.content)
        text_to_analyze = "\n".join(filter(None, final_content))
        if not text_to_analyze:
            return False, "Empty response from moray"

    elif agent_type == "squid":
        # Extract final content from squid result
        final_content = []

        # Handle dictionary structure
        if isinstance(result, dict):
            # Add any text content from the result
            if "final_results" in result:
                for res in result["final_results"]:
                    if isinstance(res, dict) and "specialist_responses" in res:
                        if "stock_analyst_plottable" in res["specialist_responses"]:
                            final_content.append(
                                res["specialist_responses"]["stock_analyst_plottable"]
                            )

            # Check for task flow data
            if "task_flow_data" in result and isinstance(
                result["task_flow_data"], dict
            ):
                if "final_compilation" in result["task_flow_data"]:
                    final_content.append(result["task_flow_data"]["final_compilation"])

            # If there's a direct response or message
            if "response" in result:
                final_content.append(str(result["response"]))
            if "message" in result:
                final_content.append(str(result["message"]))

        text_to_analyze = "\n".join(filter(None, final_content))
        if not text_to_analyze:
            return False, "Empty response from squid"

    # Check for errors using LLM
    error_check_prompt = error_check_prompt_template.format(
        text_to_analyze=text_to_analyze
    )
    response = llm.invoke(error_check_prompt)
    error_response = response.content.lower().strip()

    if error_response.startswith("false"):
        # Extract error reason if provided
        error_parts = error_response.split(":", 1)
        error_reason = (
            error_parts[1].strip() if len(error_parts) > 1 else "Unknown error"
        )
        logger.warning(f"Error detected in {agent_type} response: {error_reason}")
        return False, error_reason

    # If LLM says 'true', no error was detected
    logger.info(f"No error detected in {agent_type} response")
    return True, ""
