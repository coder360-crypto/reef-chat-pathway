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

class NewsInput(BaseModel):
    """
    Pydantic model for validating news request input.
    
    Args:
        ticker (str): Stock ticker symbol
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch the latest news for")

@tool(args_schema=NewsInput)
def get_stock_news(ticker: str):
    """
    Fetch the latest news articles for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol to fetch news for
        
    Returns:
        list: List of news article dictionaries containing headlines and details
    """
    
    # Create a Ticker object for the given symbol
    stock = yf.Ticker(ticker)
    # Fetch news articles from Yahoo Finance
    news = stock.news
    
    return news
