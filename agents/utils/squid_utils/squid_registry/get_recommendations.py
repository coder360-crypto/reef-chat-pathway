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

# Define an enum for the type of recommendation information
class RecommendationType(str, Enum):
    # Available types of stock recommendations that can be fetched
    recommendations = "recommendations"
    recommendations_summary = "recommendations_summary"
    upgrades_downgrades = "upgrades_downgrades"

# Input schema for the recommendation tool
class RecommendationsInput(BaseModel):
    """Schema defining the input structure for stock recommendations.
    
    Args:
        BaseModel: Pydantic base model for input validation
    """
    ticker: str = Field(..., description="The ticker symbol of the stock to fetch recommendations for")
    recommendation_type: RecommendationType = Field(..., description="The type of recommendation information to fetch")

@tool(args_schema=RecommendationsInput)
def get_recommendations(ticker: str, recommendation_type: RecommendationType):
    """Fetch the specified recommendation information for a given ticker symbol.
    
    Args:
        ticker (str): The stock ticker symbol to fetch data for
        recommendation_type (RecommendationType): Type of recommendation data to retrieve
        
    Returns:
        dict: Dictionary containing the requested recommendation data or error message
    """
    
    # Initialize yfinance Ticker object
    stock = yf.Ticker(ticker)
    
    # Fetch the appropriate recommendation information based on the input
    if recommendation_type == RecommendationType.recommendations:
        return stock.recommendations.to_dict()
    elif recommendation_type == RecommendationType.recommendations_summary:
        return stock.recommendations_summary.to_dict()
    elif recommendation_type == RecommendationType.upgrades_downgrades:
        return stock.upgrades_downgrades.to_dict()
    
    # Return error if invalid recommendation type is provided
    return {"error": "Invalid recommendation type selected."}
