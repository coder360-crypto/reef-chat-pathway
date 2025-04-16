# Import required libraries
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# Add parent directories to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain.tools import tool
from pydantic import BaseModel, Field
from enum import Enum
import yfinance as yf


# Pydantic model for validating input parameters
class SharesCountInput(BaseModel):
    """
    Pydantic model for validating share count input parameters.

    Args:
        ticker (str): The ticker symbol of the stock to fetch share count for
        start (str): The start date for the date range (YYYY-MM-DD)
        end (str): The end date for the date range (YYYY-MM-DD). Defaults to None.
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch share count for")
    start: str = Field(..., description="The start date for the date range (YYYY-MM-DD)")
    end: str = Field(None, description="The end date for the date range (YYYY-MM-DD). Defaults to None.")


@tool(args_schema=SharesCountInput)
def get_shares_count(ticker: str, start: str, end: str = None):
    """
    Fetch the number of shares outstanding over a specified date range.

    Args:
        ticker (str): Stock ticker symbol
        start (str): Start date in YYYY-MM-DD format
        end (str, optional): End date in YYYY-MM-DD format. Defaults to None.

    Returns:
        dict: Dictionary containing historical shares count data
    """
    
    # Initialize yfinance Ticker object
    stock = yf.Ticker(ticker)
    # Fetch historical shares count data
    shares_count = stock.get_shares_full(start=start, end=end)
    
    # Convert DataFrame to dictionary for JSON serialization
    shares_count_dict = shares_count.to_dict()
    
    return shares_count_dict
