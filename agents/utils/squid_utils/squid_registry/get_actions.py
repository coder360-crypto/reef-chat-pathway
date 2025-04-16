# Import required libraries
import os
import sys
from dotenv import load_dotenv

load_dotenv()
# Add parent directories to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
import yfinance as yf


# Define an enum for the type of action to fetch
class ActionType(str, Enum):
    """Enum class representing different types of stock actions that can be fetched.
    
    Args:
        str (str): Base string type for enum values
        
    Returns:
        ActionType: Enum instance representing action type
    """
    actions = "actions"
    dividends = "dividends"
    splits = "splits"


# Pydantic model for input validation
class ActionInput(BaseModel):
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch actions for")
    action_type: ActionType = Field(..., description="The type of action to fetch (actions, dividends, or splits)")

# TODO: check if action_type `actions` is used correctly
@tool(args_schema=ActionInput)
def get_stock_actions(ticker: str, action_type: ActionType):
    """Fetch stock actions such as dividends, splits, or both for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol to fetch data for
        action_type (ActionType): Type of action to fetch (actions/dividends/splits)
        
    Returns:
        dict: Dictionary containing the requested stock action data or error message
    """
    
    # Initialize yfinance Ticker object
    stock = yf.Ticker(ticker)
    
    # Return appropriate data based on action type
    if action_type == ActionType.actions:
        return stock.actions.to_dict()
    elif action_type == ActionType.dividends:
        return stock.dividends.to_dict()
    elif action_type == ActionType.splits:
        return stock.splits.to_dict()
    
    # Return error if invalid action type
    return {"error": "Invalid action type selected."}
