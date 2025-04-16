# Import required libraries
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Add parent directories to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pydantic import BaseModel, Field
from langchain.tools import tool
import yfinance as yf


# Pydantic model for validating input parameters
class OptionsInput(BaseModel):
    """
    Pydantic model for validating options input parameters.

    Args:
        ticker (str): The ticker symbol of the stock

    Returns:
        OptionsInput: A validated options input model
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch options expiration dates for")


@tool(args_schema=OptionsInput)
def get_options_expiration_dates(ticker: str):
    """
    Fetch the available options expiration dates for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol to fetch options data for
        
    Returns:
        dict: Dictionary containing list of options expiration dates
            - options_expiration_dates (List[str]): Available expiration dates
    """
    
    # Create a Ticker object for the given symbol
    stock = yf.Ticker(ticker)
    # Get list of available expiration dates
    options_dates = stock.options
    
    # Return dates in a structured format
    return {"options_expiration_dates": options_dates}
