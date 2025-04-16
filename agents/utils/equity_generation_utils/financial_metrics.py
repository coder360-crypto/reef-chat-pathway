import os
import sys
from typing import Dict, Optional, Union

import yfinance as yf

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AgentsConfig as Config
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain.callbacks import OpenAICallbackHandler
from utils.equity_generation_utils.cost_tracker import CostTracker

config = Config()


def company_analysis(company_name):
    """
    Searches for company's stock ticker symbol using Tavily API
    
    Args:
        company_name (str): Name of the company to search for
        
    Returns:
        str: Search results containing ticker information
    """
    tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)

    # Execute search query
    response = tavily_client.search(
        f"What is the stock ticker symbol for {company_name} for Yahoo Finance?"
    )
    results = str(response["results"])

    return results


def get_stock_ticker(company_name):
    """Get stock ticker symbol using LLM"""
    context = company_analysis(company_name)  # Get context from Tavily
    cost_tracker = CostTracker()
    callback = OpenAICallbackHandler()

    # Prepare messages for LLM
    messages = [
        {
            "role": "system",
            "content": f'Given the company name "{company_name}"\n\nContext:\n{context}\n\nReturn ONLY the ticker symbol, nothing else.',
        }
    ]

    # Initialize OpenAI LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=config.OPENAI_API_KEY,
        callbacks=[callback],  # Add callback here
    )

    response = llm.invoke(messages).content

    # Track costs after the call
    cost_tracker.add_usage(
        input_tokens=callback.prompt_tokens,
        output_tokens=callback.completion_tokens,
        cost=callback.total_cost,
    )

    return response


def get_financial_metrics(
    ticker_symbol: str, fiscal_year: Optional[int] = None
) -> Dict[str, Union[float, str]]:
    """
    Get key financial metrics for a given company using yfinance

    Args:
        ticker_symbol (str): Stock ticker symbol
        fiscal_year (int, optional): Specific fiscal year to analyze

    Returns:
        Dict containing financial metrics with values formatted in millions/billions
    """
    try:
        # Initialize ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Get financial data
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        income_stmt = ticker.income_stmt

        metrics = {}

        def format_value(value):
            """Helper to format large numbers into B/M"""
            if value is None:
                return None
            if abs(value) >= 1e9:
                return f"{value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.2f}M"
            return f"{value:.2f}"

        # Company Info
        metrics["Sector"] = info.get("sector", "N/A")
        metrics["Industry"] = info.get("industry", "N/A")
        metrics["Country"] = info.get("country", "N/A")
        metrics["Exchange"] = info.get("exchange", "N/A")

        # Current Market Price
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price is not None:
            metrics["Current Price"] = f"₹{current_price:.2f}"

        # Target Price - Get analyst target price
        target_price = info.get("targetMeanPrice")
        if target_price is not None:
            metrics["Target Price"] = f"₹{target_price:.2f}"

        target_period = "12 Months"
        if target_price is not None:
            metrics["Target Period"] = target_period

        # Market Capitalization
        market_cap = info.get("marketCap")
        if market_cap is not None:
            metrics["Market Capitalization"] = format_value(market_cap)

        # If fiscal year is specified, get data for that year
        if fiscal_year and fiscal_year in financials.columns:
            year_data = financials[fiscal_year]
            balance_sheet_data = balance_sheet[fiscal_year]
            income_data = (
                income_stmt[fiscal_year] if fiscal_year in income_stmt.columns else None
            )
        else:
            # Get most recent year
            year_data = financials.iloc[:, 0]
            balance_sheet_data = balance_sheet.iloc[:, 0]
            income_data = income_stmt.iloc[:, 0] if not income_stmt.empty else None

        # Gross Debt (Total Debt)
        gross_debt = balance_sheet_data.get("Total Debt") or balance_sheet_data.get(
            "Long Term Debt"
        )
        if gross_debt is not None:
            metrics["FY24 Gross Debt"] = format_value(gross_debt)

        # Revenues
        revenue = year_data.get("Total Revenue") or (
            income_data.get("Total Revenue") if income_data is not None else None
        )
        if revenue is not None:
            metrics["Revenues"] = format_value(revenue)

        # EBITDA - Try multiple possible column names
        ebitda = year_data.get("EBITDA") or (
            income_data.get("EBITDA") if income_data is not None else None
        )
        if ebitda is None:
            # Calculate EBITDA if not directly available
            operating_income = year_data.get("Operating Income") or (
                income_data.get("Operating Income") if income_data is not None else None
            )
            depreciation = year_data.get("Depreciation & Amortization") or (
                income_data.get("Depreciation & Amortization")
                if income_data is not None
                else None
            )
            if operating_income is not None and depreciation is not None:
                ebitda = operating_income + depreciation

        if ebitda is not None:
            metrics["EBITDA"] = format_value(ebitda)

            # EBITDA Margin - only calculate if we have both EBITDA and Revenue
            if revenue is not None and revenue != 0:
                metrics["EBITDA margin (%)"] = round((ebitda / revenue) * 100, 2)

        # Net Profit
        net_profit = year_data.get("Net Income") or (
            income_data.get("Net Income") if income_data is not None else None
        )
        if net_profit is not None:
            metrics["Net Profit"] = format_value(net_profit)

        # EPS - Try multiple approaches
        eps = info.get("trailingEPS")
        if eps is None and net_profit is not None:
            # Calculate EPS manually if not available
            shares_outstanding = balance_sheet_data.get(
                "Common Stock Shares Outstanding"
            ) or info.get("sharesOutstanding")
            if shares_outstanding is not None:
                eps = round(net_profit / shares_outstanding, 2)

        if eps is not None:
            metrics["EPS (Rs)"] = eps

            # P/E Ratio - only calculate if we have valid EPS
            if eps != 0:
                current_price = info.get("currentPrice") or info.get(
                    "regularMarketPrice"
                )
                if current_price is not None:
                    metrics["P/E (x)"] = round(current_price / eps, 2)

        # EV/EBITDA
        enterprise_value = info.get("enterpriseValue")
        if enterprise_value is not None and ebitda is not None and ebitda != 0:
            metrics["EV/EBITDA (x)"] = round(enterprise_value / ebitda, 2)

        # RoCE (Return on Capital Employed)
        ebit = year_data.get("EBIT") or (
            income_data.get("EBIT") if income_data is not None else None
        )
        if ebit is None:
            # Calculate EBIT if not available
            ebit = year_data.get("Operating Income") or (
                income_data.get("Operating Income") if income_data is not None else None
            )

        total_assets = balance_sheet_data.get("Total Assets")
        current_liabilities = balance_sheet_data.get("Total Current Liabilities")

        if all(x is not None for x in [ebit, total_assets, current_liabilities]):
            capital_employed = total_assets - current_liabilities
            if capital_employed != 0:
                metrics["RoCE (%)"] = round((ebit / capital_employed) * 100, 2)

        # RoE (Return on Equity)
        if net_profit is not None:
            shareholders_equity = balance_sheet_data.get(
                "Total Stockholder Equity"
            ) or balance_sheet_data.get("Stockholders Equity")
            if shareholders_equity is not None and shareholders_equity != 0:
                metrics["RoE (%)"] = round((net_profit / shareholders_equity) * 100, 2)

        return metrics

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return {}


if __name__ == "__main__":
    # Example usage
    ticker = get_stock_ticker("Larsen & Toubro")
    metrics = get_financial_metrics(ticker)

    print(f"\nFinancial Metrics for {ticker}:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value}")
