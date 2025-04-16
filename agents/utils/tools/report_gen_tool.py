import yfinance as yf
from typing import Dict, Union, Optional

def get_financial_metrics(ticker_symbol: str, fiscal_year: Optional[int] = None) -> Dict[str, Union[float, str]]:
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
        
        # Get financial data from yfinance API
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        income_stmt = ticker.income_stmt  # Added income statement
        
        metrics = {}
        
        def format_value(value):
            """
            Helper to format large numbers into B/M
            
            Args:
                value (float): Number to format
                
            Returns:
                str: Formatted string with B/M suffix
            """
            if value is None:
                return None
            if abs(value) >= 1e9:
                return f"{value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.2f}M"
            return f"{value:.2f}"
        
        # Market Capitalization
        market_cap = info.get('marketCap')
        if market_cap is not None:
            metrics['Market Cap'] = format_value(market_cap)
        
        # If fiscal year is specified, get data for that year
        if fiscal_year and fiscal_year in financials.columns:
            year_data = financials[fiscal_year]
            balance_sheet_data = balance_sheet[fiscal_year]
            income_data = income_stmt[fiscal_year] if fiscal_year in income_stmt.columns else None
        else:
            # Get most recent year
            year_data = financials.iloc[:, 0]
            balance_sheet_data = balance_sheet.iloc[:, 0]
            income_data = income_stmt.iloc[:, 0] if not income_stmt.empty else None
            
        # Calculate Gross Debt
        gross_debt = balance_sheet_data.get('Total Debt') or balance_sheet_data.get('Long Term Debt')
        if gross_debt is not None:
            metrics['Gross Debt'] = format_value(gross_debt)
        
        # Extract Revenue figures
        revenue = year_data.get('Total Revenue') or (income_data.get('Total Revenue') if income_data is not None else None)
        if revenue is not None:
            metrics['Revenue'] = format_value(revenue)
        
        # EBITDA - Try multiple possible column names
        ebitda = year_data.get('EBITDA') or (income_data.get('EBITDA') if income_data is not None else None)
        if ebitda is None:
            # Calculate EBITDA if not directly available
            operating_income = year_data.get('Operating Income') or (income_data.get('Operating Income') if income_data is not None else None)
            depreciation = year_data.get('Depreciation & Amortization') or (income_data.get('Depreciation & Amortization') if income_data is not None else None)
            if operating_income is not None and depreciation is not None:
                ebitda = operating_income + depreciation
        
        if ebitda is not None:
            metrics['EBITDA'] = format_value(ebitda)
            
            # EBITDA Margin - only calculate if we have both EBITDA and Revenue
            if revenue is not None and revenue != 0:
                metrics['EBITDA Margin'] = f"{round((ebitda / revenue) * 100, 2)}%"
        
        # Net Profit
        net_profit = year_data.get('Net Income') or (income_data.get('Net Income') if income_data is not None else None)
        if net_profit is not None:
            metrics['Net Profit'] = format_value(net_profit)
        
        # EPS - Try multiple approaches
        eps = info.get('trailingEPS')
        if eps is None and net_profit is not None:
            # Calculate EPS manually if not available
            shares_outstanding = balance_sheet_data.get('Common Stock Shares Outstanding') or info.get('sharesOutstanding')
            if shares_outstanding is not None:
                eps = round(net_profit / shares_outstanding, 2)
        
        if eps is not None:
            metrics['EPS'] = f"${eps:.2f}"
            
            # P/E Ratio - only calculate if we have valid EPS
            if eps != 0:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if current_price is not None:
                    metrics['PE Ratio'] = f"{round(current_price / eps, 2)}x"
        
        # EV/EBITDA
        enterprise_value = info.get('enterpriseValue')
        if enterprise_value is not None and ebitda is not None and ebitda != 0:
            metrics['EV/EBITDA'] = f"{round(enterprise_value / ebitda, 2)}x"
        
        # RoCE (Return on Capital Employed)
        ebit = year_data.get('EBIT') or (income_data.get('EBIT') if income_data is not None else None)
        if ebit is None:
            # Calculate EBIT if not available
            ebit = year_data.get('Operating Income') or (income_data.get('Operating Income') if income_data is not None else None)
            
        total_assets = balance_sheet_data.get('Total Assets')
        current_liabilities = balance_sheet_data.get('Total Current Liabilities')
        
        if all(x is not None for x in [ebit, total_assets, current_liabilities]):
            capital_employed = total_assets - current_liabilities
            if capital_employed != 0:
                metrics['RoCE'] = f"{round((ebit / capital_employed) * 100, 2)}%"
        
        # RoE (Return on Equity)
        if net_profit is not None:
            shareholders_equity = balance_sheet_data.get('Total Stockholder Equity') or balance_sheet_data.get('Stockholders Equity')
            if shareholders_equity is not None and shareholders_equity != 0:
                metrics['RoE'] = f"{round((net_profit / shareholders_equity) * 100, 2)}%"
            
        return metrics
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return {}

if __name__ == "__main__":
    metrics = get_financial_metrics('MSFT', 2024)
    print(metrics)
