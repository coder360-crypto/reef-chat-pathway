from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import yfinance as yf
import numpy as np
import asyncio

class DCFAnalysisInput(BaseModel):
    """Input for DCF Analysis.
    
    Args:
        ticker (str): The stock symbol to analyze
        discount_rate (float): Discount rate for calculations
        projection_years (int): Number of years to project
        terminal_growth_rate (float): Terminal growth rate for valuation
    """
    ticker: str = Field(description="The stock symbol to analyze (e.g., AAPL, MSFT)")
    discount_rate: float = Field(default=0.10, description="Discount rate (default: 10%)")
    projection_years: int = Field(default=5, description="Number of years to project")
    terminal_growth_rate: float = Field(default=0.02, description="Terminal growth rate (default: 2%)")

class DCFAnalysisTool(BaseTool):
    """Tool that performs DCF analysis using yfinance data.
    
    Args:
        name (str): Name of the tool
        description (str): Description of the tool's functionality
        args_schema (Optional[type[BaseModel]]): Schema for input validation
    """
    name: str = "dcf_analysis"
    description: str = "DCF analysis tool for company valuation using historical growth rates. Returns Current Price, Latest FCF, Growth Rate, DCF Value and Discount Rate"
    args_schema: Optional[type[BaseModel]] = DCFAnalysisInput

    def normalize(self, value):
        """Convert large numbers to billions/millions.
        
        Args:
            value (float): Number to normalize
            
        Returns:
            tuple[float, str]: Normalized value and unit suffix ('B', 'M', or '')
        """
        # Convert to billions if value >= 1B
        if value >= 1e9:
            return value / 1e9, 'B'
        # Convert to millions if value >= 1M
        elif value >= 1e6:
            return value / 1e6, 'M'
        return value, ''

    def calculate_growth_rate(self, fcf_values):
        """Calculate conservative growth rate from historical FCF."""
        # Use default 5% growth if insufficient data
        if len(fcf_values) < 2:
            return 0.05
        # Calculate Compound Annual Growth Rate (CAGR)
        cagr = (fcf_values[-1] / fcf_values[0]) ** (1 / (len(fcf_values) - 1)) - 1
        return max(min(cagr, 0.20), 0.02)  # Cap between 2% and 20%

    def _run(self, ticker: str, discount_rate: float = 0.10,
             projection_years: int = 5, terminal_growth_rate: float = 0.02) -> str:
        """Sync implementation - redirects to async
        
        Args:
            ticker (str): Stock symbol
            discount_rate (float): Discount rate for calculations
            projection_years (int): Number of years to project
            terminal_growth_rate (float): Terminal growth rate for valuation
            
        Returns:
            str: Error message (Not implemented)
            
        Raises:
            NotImplementedError: Always raises this error
        """
        raise NotImplementedError("Use async version")

    async def _arun(self, ticker: str, discount_rate: float = 0.10,
                    projection_years: int = 5, terminal_growth_rate: float = 0.02) -> str:
        try:
            # Fetch company data and current stock price
            company = yf.Ticker(ticker)
            current_price = company.info.get("currentPrice", 0)
            # Get historical Free Cash Flow values in reverse chronological order
            fcf_values = company.cashflow.loc["Free Cash Flow"].dropna().values[::-1]
            
            if len(fcf_values) == 0:
                return f"Error: No FCF data for {ticker}"
            
            # Calculate historical growth rate and latest FCF
            growth_rate = self.calculate_growth_rate(fcf_values)
            latest_fcf = fcf_values[-1]
            # Project future FCFs using calculated growth rate
            projected_fcfs = [latest_fcf * (1 + growth_rate) ** i 
                            for i in range(1, projection_years + 1)]
            
            # Calculate terminal value using Gordon Growth Model
            terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
            
            # Calculate present value of projected FCFs
            dcf_value = sum(fcf / (1 + discount_rate) ** i 
                          for i, fcf in enumerate(projected_fcfs, start=1))
            # Add present value of terminal value
            dcf_value += terminal_value / (1 + discount_rate) ** projection_years
            
            # Format large numbers for readability
            dcf_normalized, dcf_unit = self.normalize(dcf_value)
            latest_fcf_norm, fcf_unit = self.normalize(latest_fcf)
            
            return (
                f"DCF Analysis - {ticker}\n"
                f"Current Price: ${current_price:.2f}\n"
                f"Latest FCF: ${latest_fcf_norm:.2f}{fcf_unit}\n"
                f"Growth Rate: {growth_rate*100:.1f}%\n"
                f"DCF Value: ${dcf_normalized:.2f}{dcf_unit}\n"
                f"Discount Rate: {discount_rate*100:.1f}%"
            )
            
        except Exception as e:
            return f"Error analyzing {ticker}: {str(e)}"

# Example usage
if __name__ == "__main__":
    dcf_tool = DCFAnalysisTool()
    result = asyncio.run(dcf_tool._arun("MSFT"))
    print(result)