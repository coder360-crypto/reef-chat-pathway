# Description: A tool to fetch all stock information for a given ticker symbol.
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# Add parent directories to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain.tools import tool
from pydantic import BaseModel, Field
from enum import Enum

import yfinance as yf


class StockInfoInput(BaseModel):
    """
    Pydantic model for stock information input validation.

    Args:
        ticker (str): The stock ticker symbol to look up

    Returns:
        StockInfoInput: A validated StockInfoInput instance
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch information for")


@tool(args_schema=StockInfoInput)
def get_stock_info(ticker: str) -> dict:
    """
    Fetch all stock information for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol to fetch information for

    Returns:
        dict: A dictionary containing stock information or an error message
              Success: Contains various stock metrics and information
              Error: {"error": "Unable to fetch data. Please check the ticker symbol."}
    """
    
    # Create a Ticker object using yfinance
    stock = yf.Ticker(ticker)
    # Fetch all available information for the stock
    info = stock.info
    
    # Check if the info is retrieved properly
    if not info:
        return {"error": "Unable to fetch data. Please check the ticker symbol."}
    
    # Return the complete stock information dictionary
    return info