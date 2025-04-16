import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain.tools import tool
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
import yfinance as yf

# Define the Enum for valid periods
class PeriodEnum(str, Enum):
    """
    Enumeration of valid time periods for historical stock data retrieval
    """
    one_day = '1d'
    five_days = '5d'
    one_month = '1mo'
    three_months = '3mo'
    six_months = '6mo'
    one_year = '1y'
    two_years = '2y'
    five_years = '5y'
    ten_years = '10y'
    year_to_date = 'ytd'
    max_period = 'max'

# Define the input schema with optional and enum fields
class HistoricalDataInput(BaseModel):
    """
    Schema for validating input parameters for historical data retrieval

    Args:
        ticker (str): The ticker symbol of the stock
        period (Optional[PeriodEnum]): Time period for data retrieval, defaults to one month
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch historical data for")
    period: Optional[PeriodEnum] = Field(PeriodEnum.one_month, description="The period for which to fetch historical data")

@tool(args_schema=HistoricalDataInput)
def get_historical_data(ticker: str, period: str = "1mo") -> dict:
    """Fetch historical market data for a given ticker symbol and period.
    
    Args:
        ticker (str): The stock ticker symbol to fetch data for
        period (str, optional): Time period for historical data. Defaults to "1mo"
    
    Returns:
        dict: Historical market data with dates as keys and OHLCV data as values
    """
    
    # Initialize yfinance Ticker object
    stock = yf.Ticker(ticker)
    # Fetch historical data for specified period
    history = stock.history(period=period)
    
    # Convert the DataFrame to a dictionary for easy serialization
    history_dict = history.to_dict(orient='index')
    
    return history_dict