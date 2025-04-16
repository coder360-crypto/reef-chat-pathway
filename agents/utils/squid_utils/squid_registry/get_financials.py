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


# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    """
    Enum class representing different types of financial statements.
    
    Available types:
        - income_stmt: Annual income statement
        - quarterly_income_stmt: Quarterly income statement
        - balance_sheet: Annual balance sheet
        - quarterly_balance_sheet: Quarterly balance sheet
        - cashflow: Annual cash flow statement
        - quarterly_cashflow: Quarterly cash flow statement
    """
    # Available financial statement types - annual and quarterly reports
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


# Define the input schema
class FinancialsInput(BaseModel):
    """
    Pydantic model for validating financial statement request parameters.
    
    Attributes:
        ticker (str): The stock ticker symbol
        financial_type (FinancialType): Type of financial statement to retrieve
    """
    # Pydantic model for validating input parameters
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch financials for")
    financial_type: FinancialType = Field(..., description="The type of financial statement to fetch")


@tool(args_schema=FinancialsInput)
def get_financials(ticker: str, financial_type: FinancialType):
    """
    Fetch the specified financial statement for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol to fetch data for
        financial_type (FinancialType): Type of financial statement to retrieve
        
    Returns:
        dict: Financial statement data as a dictionary, or error message if invalid type
    """
    
    # Initialize yfinance Ticker object
    stock = yf.Ticker(ticker)
    
    # Fetch the appropriate financial statement based on the input
    if financial_type == FinancialType.income_stmt:
        return stock.income_stmt.to_dict()
    elif financial_type == FinancialType.quarterly_income_stmt:
        return stock.quarterly_income_stmt.to_dict()
    elif financial_type == FinancialType.balance_sheet:
        return stock.balance_sheet.to_dict()
    elif financial_type == FinancialType.quarterly_balance_sheet:
        return stock.quarterly_balance_sheet.to_dict()
    elif financial_type == FinancialType.cashflow:
        return stock.cashflow.to_dict()
    elif financial_type == FinancialType.quarterly_cashflow:
        return stock.quarterly_cashflow.to_dict()
    
    # Return error if invalid financial type is provided
    return {"error": "Invalid financial type selected."}
