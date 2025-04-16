# Import required libraries
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Add parent directories to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain.tools import tool
from pydantic import BaseModel, Field
import yfinance as yf


# Pydantic model for validating input parameters
class OptionChainInput(BaseModel):
    """
    Pydantic model for validating option chain input parameters.

    Args:
        ticker (str): The ticker symbol of the stock
        expiration_date (str): The expiration date in YYYY-MM-DD format
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch the options chain for")
    expiration_date: str = Field(..., description="The expiration date for the options chain (format: 'YYYY-MM-DD')")


@tool(args_schema=OptionChainInput)
def get_option_chain(ticker: str, expiration_date: str):
    """
    Fetch the options chain for a given ticker symbol and expiration date.

    Args:
        ticker (str): The stock ticker symbol
        expiration_date (str): Expiration date in YYYY-MM-DD format

    Returns:
        dict: Dictionary containing:
            - calls (dict): Call options data
            - puts (dict): Put options data
            - error (str): Error message if any error occurs
    """
    
    # Create a Ticker object for the specified stock
    stock = yf.Ticker(ticker)
    
    # Check if the expiration date is valid
    if expiration_date not in stock.options:
        return {"error": f"No options available for the date {expiration_date}. Please choose a valid expiration date."}
    
    # If the date is valid, fetch the option chain
    try:
        # Get both calls and puts data for the specified date
        option_chain = stock.option_chain(expiration_date)
        # Convert DataFrame to dictionary for easier handling
        calls_dict = option_chain.calls.to_dict(orient='index')
        puts_dict = option_chain.puts.to_dict(orient='index')
        
        # Return both calls and puts data
        return {
            "calls": calls_dict,
            "puts": puts_dict
        }
    
    except Exception as e:
        return {"error": f"An error occurred while fetching the options chain: {str(e)}"}
