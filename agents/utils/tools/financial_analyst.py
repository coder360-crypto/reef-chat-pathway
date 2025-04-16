import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import logging
from typing import List, Optional

import finnhub
import pandas as pd
import requests
import yfinance as yf
from config import AgentsConfig as Config
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.schema import OutputParserException
from langchain_community.agent_toolkits.polygon.toolkit import PolygonToolkit
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool, tool
from utils.tools.search_tools import SerperSearchTool, TavilySearchTool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()
finnhub_client = finnhub.Client(api_key=config.FINHUB_API_KEY)
polygon = PolygonAPIWrapper(polygon_api_key=config.POLYGON_API_KEY)
polygon_toolkit = PolygonToolkit.from_polygon_api_wrapper(polygon).get_tools()


def handle_parsing_error(error: OutputParserException) -> str:
    """Custom error handler for parsing errors"""
    return f"Error parsing output: {str(error)}. Please use a different tool or rephrase your request if required."


@tool
def search_symbol(query: str):
    """
    Searches for a symbol and returns its details.

    Args:
    - query (str): The symbol to search for.
    - limit (int, optional): Number of results to return. Defaults to 3.

    Returns:
    - dict: A dictionary containing the symbol's details.
    """
    try:
        return SerperSearchTool().search(f"What is the stock symbol for {query}")
    except Exception as e:
        logger.info(f"Error fetching symbol for {query} using Serper: {str(e)}")
        return TavilySearchTool(max_results=3, search_depth="basic").search(
            f"What is the stock symbol for {query}"
        )


def _make_request(params: dict) -> dict:
    """Make request to Alpha Vantage API"""
    try:
        params["apikey"] = config.ALPHAVANTAGE_API_KEY
        response = requests.get("https://www.alphavantage.co/query", params=params)
        return response.json()
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


def _make_request_polygon(params: dict) -> dict:
    """Make request to Polygon API"""
    try:
        url = f"https://api.polygon.io/v1/indicators/{params['function']}/{params['ticker']}?timespan=day&adjusted=true&window={params['window']}&series_type={params['series_type']}&order=desc&limit={params['limit']}&apiKey={config.POLYGON_API_KEY}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


def _make_request_financial_datasets(function: str, params: dict) -> dict:
    """Make request to Financial Datasets API"""
    try:
        url = f"https://api.financialdatasets.ai/financials/{function}"
        headers = {"X-API-KEY": config.FINANCIAL_DATASETS_API_KEY}
        response = requests.request("GET", url, headers=headers, params=params)
        return response.json()
    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}


class TechnicalIndicatorAnalyst:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.base_url = "https://www.alphavantage.co/query"
        self.api_key = config.ALPHAVANTAGE_API_KEY

    @tool
    def get_sma(
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ):
        """
        Fetches the Simple Moving Average (SMA) for a stock symbol.

        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly').
            time_period (int): Number of data points to average (default 20).
            series_type (str): Price type ('close', 'open', 'high', 'low').

        """
        try:
            params = {
                "function": "sma",
                "ticker": symbol,
                "timespan": "day",
                "window": time_period,
                "series_type": series_type,
                "order": "desc",
                "limit": 10,
            }
            data = _make_request_polygon(params)
            if data["status"] == "ERROR":
                url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=sma&period={time_period}&apikey={config.FMP_API_KEY}"
                data = requests.get(url).json()[:10]
                return data
            else:
                return data
        except Exception:
            url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=sma&period={time_period}&apikey={config.FMP_API_KEY}"
            data = requests.get(url).json()[:10]
            return data

    @tool
    def get_ema(
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ):
        """
        Fetches the Exponential Moving Average (EMA) for a stock symbol.

        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly') (default 'daily').
            time_period (int): Number of data points to average (default 20).
            series_type (str): Price type ('close', 'open', 'high', 'low').

        """
        try:
            params = {
                "function": "ema",
                "ticker": symbol,
                "timespan": "day",
                "window": time_period,
                "series_type": series_type,
                "order": "desc",
                "limit": 10,
            }
            data = _make_request_polygon(params)
            if data["status"] == "ERROR":
                url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=ema&period={time_period}&apikey={config.FMP_API_KEY}"
                data = requests.get(url).json()[:10]
                return data
            else:
                return data
        except Exception:
            url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=ema&period={time_period}&apikey={config.FMP_API_KEY}"
            data = requests.get(url).json()[:10]
            return data

    @tool
    def get_rsi(
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ):
        """
        Fetches the Relative Strength Index (RSI) for a stock symbol.

        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            interval (str): Time interval ('daily', 'weekly', etc.).
            time_period (int): Number of data points to average (default 20).
            series_type (str): Price type ('close', 'open', 'high', 'low').
        """
        try:
            url = f"https://api.polygon.io/v1/indicators/rsi/{symbol}?timespan=day&adjusted=true&window={time_period}&series_type={series_type}&order=desc&limit=10&apiKey={config.POLYGON_API_KEY}"
            data = requests.get(url).json()
            if data["status"] == "ERROR":
                url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=rsi&period={time_period}&apikey={config.FMP_API_KEY}"
                data = requests.get(url).json()[:10]
                return data
            else:
                return data
        except Exception:
            url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=rsi&period={time_period}&apikey={config.FMP_API_KEY}"
            data = requests.get(url).json()[:10]
            return data

    @tool
    def get_adx(
        symbol: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ):
        """
        Fetches the Average Directional Index (ADX) for a stock symbol.

        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            interval (str): Time interval ('1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly').
            time_period (int): Number of data points to average (default 20).
            series_type (str): Price type ('close', 'open', 'high', 'low').
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{symbol}?type=adx&period={time_period}&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:10]
        except Exception as e:
            return {"error": f"Error fetching ADX for {symbol}: {str(e)}"}

    @tool
    def get_macd(
        symbol: str, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9
    ):
        """
        Fetches the Moving Average Convergence Divergence (MACD) for a stock symbol.

        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            fastperiod (int): Fast period (default 12).
            slowperiod (int): Slow period (default 26).
            signalperiod (int): Signal period (default 9).
        """
        try:
            url = f"https://api.polygon.io/v1/indicators/macd/{symbol}?timespan=day&adjusted=true&short_window={fastperiod}&long_window={slowperiod}&signal_window={signalperiod}&series_type=close&order=desc&limit=10&apiKey={config.POLYGON_API_KEY}"
            data = requests.get(url).json()
            return data
        except Exception as e:
            return {"error": f"Error fetching MACD for {symbol}: {str(e)}"}

    @tool
    def get_obv(symbol: str, interval: str = "daily", limit: Optional[int] = 7):
        """
        Fetches the On Balance Volume (OBV) for a stock symbol.

        Args:
            symbol (str): Stock ticker (e.g., 'AAPL').
            interval (str): Time interval ('daily', 'weekly', etc.).
            limit (int, optional): Number of data points to return (default 7).

        """
        try:
            params = {
                "function": "OBV",
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            }
            data = _make_request(params)
            if "Technical Analysis: OBV" in data:
                values = data["Technical Analysis: OBV"]
                dates = sorted(values.keys(), reverse=True)[:limit]
                return [
                    {"date": date, "OBV": float(values[date]["OBV"])} for date in dates
                ]
            return data
        except Exception as e:
            return {"error": f"Error fetching OBV for {symbol}: {str(e)}"}

    def get_technical_tool(self):
        # Create the agent using the tools
        agent_executor = self.create_technical_indicator_agent()

        def analyze_technical_content(query: str, context: Optional[str] = None):

            if context:
                chain_input = {
                    "query": f"{query}, Provided the following context: {context}"
                }
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "technical_indicator_analyst"},
            }

        return StructuredTool.from_function(
            name="technical_indicator_analyst",
            func=analyze_technical_content,
            description="Tool for analyzing technical indicators including SMA, EMA, RSI, and ADX.",
        )

    def create_technical_indicator_agent(self):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the user query as an advanced technical indicator analyst. Follow these comprehensive analysis guidelines:

            1. If the provided context is not relevant, ignore it or else analyze the technical indicators for it.
            2. Use the tools provided to fetch technical indicators if not provided in the context.
            3. If the query is not related to technical indicators, use the search_symbol tool to find the stock symbol.
            4. If the query is related to technical indicators, use the tools provided to fetch the technical indicators.
            
            1. TREND ANALYSIS:
               - Use multiple moving averages (SMA, EMA) to identify primary and secondary trends
               - Compare different timeframes (short-term vs long-term MAs)
               - Look for key support/resistance levels
               - Identify trend patterns (ascending, descending, sideways)

            2. MOMENTUM ANALYSIS:
               - Use RSI to identify overbought (>70) or oversold (<30) conditions
               - Confirm with Stochastic oscillator readings
               - Check for momentum divergences with price
               - Analyze volume trends alongside price movements

            3. VOLATILITY ASSESSMENT:
               - Use OBV to confirm price trends with volume
               - Look for unusual volume spikes
               - Identify potential breakout or breakdown points

            4. SIGNAL INTEGRATION:
               - Cross-reference signals from multiple indicators
               - Weight recent data more heavily than older data
               - Look for confluence of multiple signals
               - Note any conflicting indicators

            5. RISK ASSESSMENT:
               - Identify key price levels for stop losses
               - Note market conditions affecting volatility
               - Consider position sizing based on volatility
               - Highlight potential reversal points

            Provide a structured analysis that:
            1. Summarizes the overall technical picture
            2. Details specific indicator readings and their implications
            3. Notes any conflicting signals
            4. Suggests potential entry/exit points
            5. Highlights key risk levels
            """,
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create the agent using the tools
        agent = create_tool_calling_agent(
            model,
            [
                self.get_sma,
                self.get_ema,
                self.get_adx,
                self.get_rsi,
                self.get_obv,
                self.get_macd,
                search_symbol,
            ],
            prompt,
        )

        # Create the AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=[
                self.get_sma,
                self.get_ema,
                self.get_adx,
                self.get_rsi,
                self.get_obv,
                self.get_macd,
                search_symbol,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )

        return agent_executor


# Financial News Agent
class FinancialNewsAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.polygon_news = polygon_toolkit[3]

    @tool
    def get_news_sentiment(
        tickers: Optional[List[str]] = None,
        topics: str = "Finance",
        time_from: str = "20240101T0130",
        time_to: Optional[str] = None,
        sort: str = "LATEST",
        limit: int = 3,
    ):
        """
        Fetches news sentiment for specified tickers.

        Args:
            tickers (list): List of stock tickers to analyze.
            topics (str): The topic of news to analyze (default is 'Finance').
            time_from (str): Start time for fetching news (default is '20240101T0130').
            time_to (str): End time for fetching news (optional).
            sort (str): Sorting order of news (default is 'LATEST').
            limit (int): Number of news articles to return (default is 5).
        """
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": tickers,
                "topics": topics,
                "time_from": time_from,
                "time_to": time_to,
                "sort": sort,
                "limit": limit,
            }
            data = _make_request(params)
            return data
        except Exception as e:
            return {"error": f"Error fetching news sentiment for {tickers}: {str(e)}"}

    @tool
    def general_news(category: str = "general", min_id: Optional[int] = None):
        """
        Returns the general financial news for the given category and minimum id.

        Args:
            category (str): The category to get the news for (e.g., 'general', 'forex', 'crypto', 'merger').
            min_id (int): The minimum id to get the news for.
        """
        try:
            results = finnhub_client.general_news(category, min_id)
            return results[:3]
        except Exception as e:
            return {"error": f"Error fetching general news for {category}: {str(e)}"}

    @tool
    def company_news(
        symbol: str,
        _from: Optional[str] = "2024-06-01",
        to: Optional[str] = "2024-11-24",
    ):
        """
        Returns the company news for the given symbol and date range.

        Args:
            symbol (str): The symbol to get the news for.
            _from (str): The start date to get the news for.
            to (str): The end date to get the news for.
        """
        try:
            results = finnhub_client.company_news(symbol, _from, to)
            return results[:3]
        except Exception as e:
            ticker = yf.Ticker(symbol)
            if ticker.news is not None:
                return ticker.news
            else:
                url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=10&apiKey={config.POLYGON_API_KEY}"
                return requests.get(url).json()

    def get_news_tool(self):

        agent_executor = self.create_news_agent()

        def analyze_news_content(query: str, context: Optional[str] = None):
            if context:
                chain_input = {
                    "query": f"{query}, Provided the following context: {context}"
                }
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "financial_news_analyst"},
            }

        return StructuredTool.from_function(
            name="financial_news_analyst",
            func=analyze_news_content,
            description="Tool for analyzing financial news and sentiment data.",
        )

    def create_news_agent(self):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the user query as an expert financial news analyst. Follow these comprehensive analysis guidelines:
            
            1. If the provided context is not relevant, ignore it or else analyze the financial news for it.
            2. Use the tools provided to fetch financial news if not provided in the context.
            3. If the query is not related to financial news, use the search_symbol tool to find the stock symbol.
            4. If the query is related to financial news, use the tools provided to fetch the financial news.

            1. NEWS SENTIMENT ANALYSIS:
               - Analyze overall market sentiment
               - Identify key news themes and patterns
               - Weight news by source credibility and timeliness
               - Look for sentiment shifts over time

            2. COMPANY-SPECIFIC NEWS:
               - Analyze company announcements
               - Track management changes and strategic updates
               - Monitor analyst coverage and ratings changes
               - Follow product/service developments

            3. SECTOR ANALYSIS:
               - Identify sector-wide trends and news
               - Compare company performance to sector peers
               - Track regulatory changes affecting the sector
               - Monitor sector rotation patterns

            4. MARKET IMPACT ASSESSMENT:
               - Evaluate news impact on stock price
               - Analyze volume changes related to news
               - Track institutional investor reactions
               - Monitor social media sentiment

            5. RISK FACTOR ANALYSIS:
               - Identify potential risks from news
               - Track legal and regulatory developments
               - Monitor competitive threats
               - Assess market condition changes

            Provide a structured analysis that:
            1. Summarizes key news developments
            2. Analyzes sentiment trends
            3. Highlights potential market impacts
            4. Identifies key risks and opportunities
            5. Suggests areas for further monitoring
            """,
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(
            model,
            [
                self.get_news_sentiment,
                self.general_news,
                self.company_news,
                self.polygon_news,
                search_symbol,
            ],
            prompt,
        )
        return AgentExecutor(
            agent=agent,
            tools=[
                self.get_news_sentiment,
                self.general_news,
                self.company_news,
                self.polygon_news,
                search_symbol,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )


# Corporate Actions Agent
class CorporateActionsAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @tool
    def get_actions(symbol: str):
        """
        Fetches the actions for a given stock symbol.

        Args:
            symbol (str): The stock ticker (e.g., 'AAPL').

        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.actions
        except Exception:
            pass

    @tool
    def get_dividends(symbol: str):
        """
        Fetches the dividends for a given stock symbol.

        Args:
            symbol (str): The stock ticker (e.g., 'AAPL').

        """
        try:
            ticker = yf.Ticker(symbol)
            if ticker.dividends is not None:
                return ticker.dividends
            else:
                url = f"https://api.polygon.io/v3/reference/dividends?ticker={symbol}&limit=10&apiKey={config.POLYGON_API_KEY}"
                return requests.get(url).json()
        except Exception:
            url = f"https://api.polygon.io/v3/reference/dividends?ticker={symbol}&limit=10&apiKey={config.POLYGON_API_KEY}"
            return requests.get(url).json()

    @tool
    def get_splits(symbol: str):
        """
        Fetches the stock splits for a given stock symbol.

        Args:
            symbol (str): The stock ticker (e.g., 'AAPL').

        """
        try:
            ticker = yf.Ticker(symbol)
            if ticker.splits is not None:
                return ticker.splits
            else:
                url = f"https://api.polygon.io/v3/reference/splits?ticker={symbol}&limit=10&apiKey={config.POLYGON_API_KEY}"
                return requests.get(url).json()
        except Exception:
            url = f"https://api.polygon.io/v3/reference/splits?ticker={symbol}&limit=10&apiKey={config.POLYGON_API_KEY}"
            return requests.get(url).json()

    @staticmethod
    @tool
    def company_profile(symbol: str):
        """
        Returns the company profile for the given symbol.

        Args:
            symbol (str): The symbol to get the company profile for.
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception:
            results = finnhub_client.company_profile2(symbol=symbol)
            return results

    @tool
    def get_etf_profile(symbol: str):
        """
        This API returns key ETF metrics (e.g., net assets, expense ratio, and turnover), along with the corresponding ETF holdings / constituents with allocation by asset types and sectors.
        Args:
            symbol (str): The ETF symbol to get the profile for.
        """
        try:
            params = {"function": "ETF_PROFILE", "symbol": symbol}
            data = _make_request(params)
            return data
        except Exception as e:
            return {"error": f"Error fetching ETF profile for {symbol}: {str(e)}"}

    def get_corporate_actions_tool(self):

        agent_executor = self.create_corporate_actions_agent()

        def analyze_corporate_actions_content(
            query: str, context: Optional[str] = None
        ):
            if context:
                chain_input = {"query": f"query: {query}, context: {context}"}
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "corporate_actions_analyst"},
            }

        return StructuredTool.from_function(
            name="corporate_actions_analyst",
            func=analyze_corporate_actions_content,
            description="Tool for analyzing corporate actions such as dividends, stock splits, and company profile.",
        )

    def create_corporate_actions_agent(self):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the user query as an expert corporate actions analyst. Follow these comprehensive analysis guidelines:

            1. If the provided context is not relevant, ignore it or else analyze the corporate actions for it.
            2. Use the tools provided to fetch corporate actions if not provided in the context.
            3. If the query is not related to corporate actions, use the search_symbol tool to find the stock symbol.
            4. If the query is related to corporate actions, use the tools provided to fetch the corporate actions.
            
            1. DIVIDEND ANALYSIS:
               - Track dividend history and trends
               - Calculate dividend yield and payout ratios
               - Compare to sector averages
               - Assess dividend sustainability

            2. STOCK SPLIT ASSESSMENT:
               - Analyze historical split patterns
               - Evaluate market impact of splits
               - Compare with peer companies
               - Consider timing and rationale

            3. CORPORATE EVENT ANALYSIS:
               - Monitor merger and acquisition activity
               - Track spinoff and divestiture plans
               - Analyze share buyback programs
               - Evaluate management changes

            4. FINANCIAL IMPACT ASSESSMENT:
               - Calculate EPS impact
               - Evaluate balance sheet effects
               - Assess cash flow implications
               - Consider tax implications

            5. STAKEHOLDER IMPACT:
               - Analyze shareholder benefits/risks
               - Consider institutional investor reactions
               - Evaluate market sentiment impact
               - Track insider transactions

            Provide a structured analysis that:
            1. Summarizes recent corporate actions
            2. Analyzes financial implications
            3. Evaluates market impact
            4. Highlights risks and opportunities
            5. Suggests monitoring points
            """,
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(
            model,
            [
                self.get_dividends,
                self.get_splits,
                self.company_profile,
                self.get_etf_profile,
                self.get_actions,
            ],
            prompt,
        )
        return AgentExecutor(
            agent=agent,
            tools=[
                self.get_dividends,
                self.get_splits,
                self.company_profile,
                self.get_etf_profile,
                self.get_actions,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )


# Market Price & Trading Agent
class MarketPriceTradingAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.polygon_aggregates = polygon_toolkit[0]
        self.polygon_last_quote = polygon_toolkit[1]

    @tool
    def get_historical_data(
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        prepost: bool = False,
        actions: bool = True,
        auto_adjust: bool = True,
        back_adjust: bool = False,
        repair: bool = False,
        keepna: bool = False,
        proxy: Optional[str] = None,
        rounding: bool = False,
        timeout: int = 10,
        raise_errors: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches historical market data for a given stock symbol.

        Args:
            symbol (str): The stock symbol for which to retrieve historical data.
            period (str, optional): The time period for which to fetch data. Defaults to '1mo'.
            interval (str, optional): The time interval for which to fetch data. Defaults to '1d'.
            start (Optional[str], optional): The start date for fetching data. Defaults to None.
            end (Optional[str], optional): The end date for fetching data. Defaults to None.
            prepost (bool, optional): Include pre and post market data. Defaults to False.
            actions (bool, optional): Include dividends and stock splits. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the historical market data.
        """
        try:
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                start=start,
                end=end,
                prepost=prepost,
                actions=actions,
                auto_adjust=auto_adjust,
                back_adjust=back_adjust,
                repair=repair,
                keepna=keepna,
                proxy=proxy,
                rounding=rounding,
                timeout=timeout,
                raise_errors=raise_errors,
            )
            if data is not None:
                return data
            else:
                return {
                    "error": f"Error fetching historical data for {symbol}: {str(e)}"
                }
        except Exception as e:
            return {"error": f"Error fetching historical data for {symbol}: {str(e)}"}

    @tool
    def get_current_price(symbol: str):
        """
        Use this tool to get the current market price for a given stock symbol.

        Args:
            symbol (str): The stock symbol for which to retrieve the current price.
        """
        try:
            return finnhub_client.quote(symbol)
        except Exception:
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()

    @tool
    def get_top_gainers():
        """
        Fetches the top gainers in the market.

        Returns:
            dict: A dictionary containing the top gainers.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:10]
        except Exception as e:
            logger.info(f"Error fetching top gainers: {str(e)}")
            params = {"function": "TOP_GAINERS_LOSERS", "market": "us"}
            data = _make_request(params)
            return data["top_gainers"]

    @tool
    def get_top_losers():
        """
        Fetches the top losers in the market.

        Returns:
            dict: A dictionary containing the top losers.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:10]
        except Exception as e:
            logger.info(f"Error fetching top losers: {str(e)}")
            params = {"function": "TOP_GAINERS_LOSERS", "market": "us"}
            data = _make_request(params)
            return data["top_losers"]

    @tool
    def get_most_active():
        """
        Fetches the most active stocks in the market.

        Returns:
            dict: A dictionary containing the most active stocks.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/stock_market/actives?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:10]
        except Exception as e:
            logger.info(f"Error fetching most active stocks: {str(e)}")
            params = {"function": "TOP_GAINERS_LOSERS", "market": "us"}
            data = _make_request(params)
            return data["most_actively_traded"]

    @tool
    def market_status(exchange: str) -> dict:
        """
        Returns the market status for the given exchange. It gives the holiday, open/close status, and session.

        Args:
            exchange (str): The exchange to get the market status for.

        Returns:
            dict: A dictionary containing the market status.
        """
        try:
            results = finnhub_client.market_status(exchange)
            return results
        except Exception as e:
            return {"error": f"Error fetching market status for {exchange}: {str(e)}"}

    @tool
    def get_sector_performance():
        """
        Retrieves the current performance of all market sectors.

        Returns:
        list: List of sectors with their current performance metrics including changes and percentages
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/sectors-performance?apikey={config.FMP_API_KEY}"
            data = requests.get(url).json()
            return data
        except Exception as e:
            return {"error": f"Error fetching sector performance: {str(e)}"}

    @tool
    def get_historical_sector_performance(from_date: str, to_date: str):
        """
        Retrieves historical performance data for all market sectors within a specified date range.

        Args:
            from_date (str): Start date in format 'YYYY-MM-DD'
            to_date (str): End date in format 'YYYY-MM-DD'

        Returns:
            list: Historical performance data for all sectors between the specified dates
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-sectors-performance?from={from_date}&to={to_date}&apikey={config.FMP_API_KEY}"
            data = requests.get(url).json()
            return data
        except Exception as e:
            return {"error": f"Error fetching historical sector performance: {str(e)}"}

    @tool
    def company_basic_financials(symbol: str) -> dict:
        """
        Returns the Current ratio, sales per share, net margin, 52 week high, 52 week low, 50 day moving average, beta, and 10 day average volume for the given symbol.

        Args:
            symbol (str): The symbol to get the financial metrics for.

        Returns:
            dict: A dictionary containing the financial metrics.
        """
        try:
            results = finnhub_client.company_basic_financials(symbol, "all")
            return results
        except Exception:
            ticker = yf.Ticker(symbol)
            return ticker.financials

    def get_trading_tool(self):

        agent_executor = self.create_trading_agent()

        def analyze_trading_content(query: str, context: Optional[str] = None):
            if context:
                chain_input = {"query": f"query: {query}, context: {context}"}
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "trading_analyst"},
            }

        return StructuredTool.from_function(
            name="trading_analyst",
            func=analyze_trading_content,
            description="Tool for analyzing market price and trading metrics.",
        )

    def create_trading_agent(self):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user query as a market price and trading analyst, providing real-time market data and trading insights. You can handle complex queries about market trends, trading strategies, and price movements. Always provide actionable insights based on the latest market data. Don't provide any information that is not present in the context or any feedbacks or negative comments on missing documents. Always try your best to provide the most accurate by using the available tools.",
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(
            model,
            [
                self.get_top_gainers,
                self.get_top_losers,
                self.get_most_active,
                self.market_status,
                self.get_current_price,
                self.get_historical_data,
                self.company_basic_financials,
                self.get_sector_performance,
                self.get_historical_sector_performance,
            ],
            prompt,
        )
        return AgentExecutor(
            agent=agent,
            tools=[
                self.get_top_gainers,
                self.get_top_losers,
                self.get_most_active,
                self.market_status,
                self.get_current_price,
                self.get_historical_data,
                self.company_basic_financials,
                self.get_sector_performance,
                self.get_historical_sector_performance,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )


class FinancialMetricsScoresAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @tool
    def get_insider_sentiment(
        symbol: str, from_date: str = "2024-01-01", to_date: str = "2024-12-31"
    ):
        """
        Get insider sentiment data for US companies calculated using MSPR. The MSPR ranges from -100 for the most negative to 100 for the most positive which can signal price changes in the coming 30-90 days.

        Monthly Share Purchase Ratio (MSPR)

        Insider sentiment can be measured by Monthly Share Purchase Ratioᵢ,ₘ (MSPRᵢ,ₘ) for the company i in month m which is calculated as below:
        Where the PS and SS are the Purchasing Shares and Selling Shares of the company i in month m. Given that D is the total number of days in month m and all trading shares are positive numbers.
        The closer this number is to 1 (-1) the more reliable that the stock prices of the firm increase (decrease) in the next periods.
        Insider sentiment is considered positive if its corporate’s shares are under net purchasing activity (i.e. market purchases > market sells). Thus, net selling activity (i.e. market sells > market purchases) results in negative insider sentiment.

        Args:
            symbol (str): The symbol ticker to get the insider sentiment for.
        """
        try:
            return finnhub_client.stock_insider_sentiment(symbol, from_date, to_date)
        except Exception as e:
            return {"error": f"Error fetching insider sentiment for {symbol}: {str(e)}"}

    @tool
    def get_company_rating(symbol: str):
        """
        The FMP Company Rating endpoint provides a rating of a company based on its financial statements, discounted cash flow analysis, financial ratios, and intrinsic value. Investors can use this rating to get a quick overview of a company's financial health and to compare different companies.

        Args:
            symbol (str): The symbol ticker to get the company rating for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/rating/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()
        except Exception as e:
            return {"error": f"Error fetching company rating for {symbol}: {str(e)}"}

    @tool
    def get_financial_growth(symbol: str):
        """
        Returns the financial growth for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the financial scores for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/financial-growth/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]
        except Exception as e:
            return {"error": f"Error fetching financial growth for {symbol}: {str(e)}"}

    @tool
    def get_key_metrics(symbol: str):
        """
        Returns the key metrics for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the key metrics for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()
        except Exception as e:
            ticker = yf.Ticker(symbol)
            return ticker.info

    @tool
    def get_ratios(symbol: str):
        """
        Returns the ratios for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the ratios for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()
        except Exception as e:
            return {"error": f"Error fetching ratios for {symbol}: {str(e)}"}

    @tool
    def get_dcfs(symbol: str):
        """
        Returns the discounted cash flows for the given symbol.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()
        except Exception as e:
            return {
                "error": f"Error fetching discounted cash flows for {symbol}: {str(e)}"
            }
        
    @tool
    def get_shares_full(symbol: str, start: Optional[str] = None, end: Optional[str] = None):
        """
        Retrieves the full shares data for a given symbol including outstanding shares, float shares, and other share-related metrics.
        
        Args:
            symbol (str): The stock symbol to get shares data for (e.g., 'AAPL')
            start (Optional[str]): Start date in format 'YYYY-MM-DD'. If None, defaults to 1 year ago
            end (Optional[str]): End date in format 'YYYY-MM-DD'. If None, defaults to today
        
        Returns:
            dict: Dictionary containing shares data or error message if retrieval fails
        """
        try:
            ticker = yf.Ticker(symbol)
            shares_data = ticker.get_shares_full(start=start, end=end)
            
            if shares_data is not None and not shares_data.empty:
                logger.info(f"Full shares data for {symbol} retrieved from yfinance")
                return shares_data.to_dict()
            else:
                return {"error": f"No shares data available for {symbol}"}
        except Exception as e:
            logger.info(f"Error fetching shares data for {symbol}: {str(e)}")
            return {"error": f"Error fetching shares data for {symbol}: {str(e)}"}
    @tool
    def get_eps_trend(symbol: str):
        """
        Retrieves the EPS (Earnings Per Share) trend data for a given symbol.
        Returns EPS estimates for current quarter, next quarter, current year, and next year,
        showing how estimates changed over time (current, 7 days ago, 30 days ago, 60 days ago, 90 days ago).
        
        Args:
            symbol (str): The stock symbol to get EPS trend data for (e.g., 'AAPL')
        
        Returns:
            dict: Dictionary containing EPS trend data with periods:
                - 0q (Current Quarter)
                - +1q (Next Quarter)
                - 0y (Current Year)
                - +1y (Next Year)
            For each period, shows estimates from:
                - current
                - 7daysAgo
                - 30daysAgo
                - 60daysAgo
                - 90daysAgo
        """
        try:
            ticker = yf.Ticker(symbol)
            eps_data = ticker.get_eps_trend(as_dict=True)
            
            if eps_data:
                logger.info(f"EPS trend data for {symbol} retrieved from yfinance")
                return eps_data
            else:
                return {"error": f"No EPS trend data available for {symbol}"}
        except Exception as e:
            logger.info(f"Error fetching EPS trend data for {symbol}: {str(e)}")
            return {"error": f"Error fetching EPS trend data for {symbol}: {str(e)}"}
    @tool
    def get_growth_estimates(symbol: str):
        """
        Retrieves growth estimates for a company compared to its industry, sector, and market index.
        
        Args:
            symbol (str): The stock symbol to get growth estimates for (e.g., 'AAPL')
        
        Returns:
            dict: Dictionary containing growth estimates for different periods:
                - 0q (Current Quarter)
                - +1q (Next Quarter)
                - 0y (Current Year)
                - +1y (Next Year)
                - +5y (Next 5 Years)
                - -5y (Past 5 Years)
            Each period includes comparisons with:
                - stock (Company's estimates)
                - industry (Industry average)
                - sector (Sector average)
                - index (Market index)
        """
        try:
            ticker = yf.Ticker(symbol)
            growth_data = ticker.get_growth_estimates(as_dict=True)
            
            if growth_data:
                logger.info(f"Growth estimates for {symbol} retrieved from yfinance")
                return growth_data
            else:
                return {"error": f"No growth estimates available for {symbol}"}
        except Exception as e:
            logger.info(f"Error fetching growth estimates for {symbol}: {str(e)}")
            return {"error": f"Error fetching growth estimates for {symbol}: {str(e)}"}

    def get_financial_metrics_tool(self):

        agent_executor = self.create_financial_metrics_agent()

        def analyze_financial_metrics_content(
            query: str, context: Optional[str] = None
        ):
            if context:
                chain_input = {"query": f" {query}, Provided context: {context}"}
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "financial_metrics_analyst"},
            }

        return StructuredTool.from_function(
            name="financial_metrics_analyst",
            func=analyze_financial_metrics_content,
            description="Tool for analyzing financial metrics and scores.",
        )

    def create_financial_metrics_agent(self):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user query as a financial metrics and scores analyst, providing real-time financial metrics and scores. You can handle complex queries about financial metrics and scores. Always provide actionable insights based on the latest financial metrics and scores. Don't provide any information that is not present in the context or any feedbacks or negative comments on missing documents. Always try your best to provide the most accurate by using the available tools.",
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(
            model,
            [
                self.get_financial_growth,
                self.get_key_metrics,
                self.get_ratios,
                self.get_dcfs,
                self.get_company_rating,
                self.get_insider_sentiment,
                self.get_shares_full,
                self.get_eps_trend,
                self.get_growth_estimates,
            ],
            prompt,
        )
        return AgentExecutor(
            agent=agent,
            tools=[
                self.get_financial_growth,
                self.get_key_metrics,
                self.get_ratios,
                self.get_dcfs,
                self.get_company_rating,
                self.get_insider_sentiment,
                self.get_shares_full,
                self.get_eps_trend,
                self.get_growth_estimates,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )


# Financial Statement Analysis Agent
class FinancialStatementAnalysisAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.polygon_financials = polygon_toolkit[2]

    @tool
    def get_key_metrics(symbol: str):
        """
        Returns the key metrics for the given symbol.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}?apikey={config.FMP_API_KEY}"
            return requests.get(url).json()
        except Exception as e:
            ticker = yf.Ticker(symbol)
            return ticker.info

    @tool
    def get_income_statement(symbol: str):
        """
        Returns the income statement for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the income statement for.
        """
        try:
            ticker = yf.Ticker(symbol)

            data = ticker.income_stmt
            if data is not None:
                logger.info(f"Income statement for {symbol} retrieved from yfinance")
                return data
            else:
                url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
                return requests.get(url).json()[:3]
        except Exception as e:
            url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]

    @tool
    def get_balance_sheet(symbol: str):
        """
        Returns the balance sheet for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the balance sheet for.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.balance_sheet
            if data is not None:
                logger.info(f"Balance sheet for {symbol} retrieved from yfinance")
                return data
            else:
                url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
                return requests.get(url).json()[:3]
        except Exception as e:
            url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]

    @tool
    def financials_reported(symbol: str):
        """
        Returns the financials reported for the given symbol. Use this only to get the financials of public companies for year 2023.

        Args:
            symbol (str): The symbol to get the financials for (e.g., 'AAPL').

        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.financials
            if data is not None:
                logger.info(f"Financials reported for {symbol} retrieved from yfinance")
                return data
            else:
                results = finnhub_client.financials_reported(symbol=symbol)
                # Sort reports by date and take only the 3 most recent ones
                sorted_data = sorted(
                    results["data"], key=lambda x: x["endDate"], reverse=True
                )[0]

            return {"cik": results["cik"], "data": sorted_data}
        except Exception:
            results = finnhub_client.financials_reported(symbol=symbol)
            # Sort reports by date and take only the 3 most recent ones
            sorted_data = sorted(
                results["data"], key=lambda x: x["endDate"], reverse=True
            )[0]

            return {"cik": results["cik"], "data": sorted_data}

    @tool
    def company_filings(symbol: str, _from: str = "24-06-01", to: str = "24-11-24"):
        """
        Returns the company filings of public companies for the given symbol and date range.

        Args:
        - symbol (str): The symbol to get the filings for.
        - _from (str): The start date to get the filings for. Use "24-06-01" as default if no start date is specified.
        - to (str): The end date to get the filings for. Use "24-11-24" as default if no end date is specified.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.sec_filings
            if data is not None:
                logger.info(f"Filings for {symbol} retrieved from yfinance")
                return data
            else:
                results = finnhub_client.filings(symbol, _from, to)
                return results[:10]
        except Exception:
            results = finnhub_client.filings(symbol, _from, to)
            return results[:10]

    @tool
    def company_earnings(symbol: str, limit: int = 4):
        """
        Returns the earnings data of public companies for the given symbol and limit.

        Args:
        - symbol (str): The symbol to get the earnings for.
        - limit (int): The number of earnings to retrieve. Use 4 as default if no limit is specified.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.earnings
            if data is not None:
                logger.info(f"Earnings for {symbol} retrieved from yfinance")
                return data
            else:
                results = finnhub_client.company_earnings(symbol, limit=limit)
                return results[:10]
        except Exception as e:
            logger.info(f"Error fetching earnings data for {symbol}: {str(e)}")
            params = {"limit": 10, "ticker": symbol}
            data = _make_request_financial_datasets("segmented-revenues", params)
            return data

    @tool
    def get_cash_flow(symbol: str):
        """
        Returns the cash flow for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the cash flow for.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.cash_flow
            if data is not None:
                logger.info(f"Cash flow for {symbol} retrieved from yfinance")
                return data
            else:
                params = {"limit": 10, "ticker": symbol}
            data = _make_request_financial_datasets("cash-flow-statements", params)

        except Exception as e:
            logger.info(f"Error fetching cash flow for {symbol}: {str(e)}")
            url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]

    @tool
    def get_cash_flow_growth(symbol: str):
        """
        Returns the cash flow growth for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the cash flow growth for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement-growth/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]
        except Exception as e:
            logger.info(f"Error fetching cash flow growth for {symbol}: {str(e)}")
            return {"error": f"Error fetching cash flow growth for {symbol}: {str(e)}"}

    @tool
    def get_income_statement_growth(symbol: str):
        """
        Returns the income statement growth for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the income statement growth for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/income-statement-growth/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]
        except Exception as e:
            logger.info(
                f"Error fetching income statement growth for {symbol}: {str(e)}"
            )
            return {
                "error": f"Error fetching income statement growth for {symbol}: {str(e)}"
            }

    @tool
    def get_balance_sheet_growth(symbol: str):
        """
        Returns the balance sheet growth for the given symbol.

        Args:
            symbol (str): The symbol ticker to get the balance sheet growth for.
        """
        try:
            url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-growth/{symbol}?period=annual&apikey={config.FMP_API_KEY}"
            return requests.get(url).json()[:3]
        except Exception as e:
            logger.info(f"Error fetching balance sheet growth for {symbol}: {str(e)}")
            return {
                "error": f"Error fetching balance sheet growth for {symbol}: {str(e)}"
            }

    def get_financial_statement_tool(self):

        agent_executor = self.create_financial_statement_agent()

        def analyze_financial_statement_content(
            query: str, context: Optional[str] = None
        ):
            if context:
                chain_input = {
                    "query": f"query: {query}, Perform analysis on the following context: {context}"
                }
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "financial_statement_analyst"},
            }

        return StructuredTool.from_function(
            name="financial_statement_analyst",
            func=analyze_financial_statement_content,
            description="Tool for analyzing financial statements including balance sheets, income, and cash flows.",
        )

    def create_financial_statement_agent(self):
        model = self.llm
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the user query as a advanced financial statement analyst, to analyze the gathered financial data or to provide deep insights on company financials. Analyze complex financial queries and each word in the query and provide detailed comprehensive analysis using the thought behind the query. If the query is complex, break it down into manageable parts. Don't use balance_tool, income_tool, cash_flow_tool if the retrieved context contains required financial data.
             Follow these comprehensive analysis as per requirements:
             
             1. Use the get_balance_sheet, get_income_statement, get_cash_flow tools if the context does not contain the following statements.
             2. Use as many tools as required to get the financial data and perform detailed analysis if required.
             3. Analyze all relevant numbers in great detail.
             4. Always use one of this formula based on the available information for calculating quick ratio : Quick Ratio = (Cash and Cash Equivalents + Short Term Investments + Receivables) / Total Current Liabilities or Quick Ratio = (Total Current Assets - Raw materials and supplies - Work in process and finish goods) / Total Current Liabilities. 
             5. Current Ratio = Current Assets / Current Liabilities.
             6. Break down complex queries into specific analytical components
             7. Perform thorough quantitative and qualitative analysis
             8. Compare historical comparisons and industry benchmarks whenever mentioned
             9. Identify key financial metrics and ratios relevant to the query
             10. Analyze year-over-year changes and growth rates
             11. Return the output in markdown format with tables if required.
            

             """,
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(
            model,
            [
                self.get_income_statement,
                self.get_balance_sheet,
                self.financials_reported,
                self.company_filings,
                self.company_earnings,
                self.get_cash_flow,
                self.get_cash_flow_growth,
                self.get_income_statement_growth,
                self.get_balance_sheet_growth,
                self.get_key_metrics,
                self.polygon_financials,
            ],
            prompt,
        )
        return AgentExecutor(
            agent=agent,
            tools=[
                self.get_income_statement,
                self.get_balance_sheet,
                self.financials_reported,
                self.company_filings,
                self.company_earnings,
                self.get_cash_flow,
                self.get_cash_flow_growth,
                self.get_income_statement_growth,
                self.get_balance_sheet_growth,
                self.get_key_metrics,
                self.polygon_financials,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )


# Macroeconomic Analysis Agent
class MacroeconomicAnalysisAgent:

    def __init__(self, llm):
        self.llm = llm

    @tool
    def get_real_gdp(interval: str = "annual"):
        """
        Fetches the Real GDP for the specified interval.

        Args:
            interval (str): The interval for fetching GDP data (default is 'annual').

        """
        try:
            params = {"function": "REAL_GDP", "interval": interval}
            data = _make_request(params)
            return data["data"][:5]
        except Exception as e:
            return {"error": f"Error fetching real GDP data: {str(e)}"}

    @tool
    def get_real_gdp_per_capita():
        """
        Fetches the Real GDP per capita.

        Returns:
            dict: The Real GDP per capita data.
        """
        try:
            params = {"function": "REAL_GDP_PER_CAPITA"}
            data = _make_request(params)["data"][:5]
            return data
        except Exception as e:
            return {"error": f"Error fetching real GDP per capita data: {str(e)}"}

    @tool
    def get_treasury_yield(interval: str = "monthly", maturity: str = "10year"):
        """
        Fetches the Treasury yield for the specified interval and maturity.

        Args:
            interval (str): The interval for fetching yield data (default is 'monthly').
            maturity (str): The maturity period (default is '10year').

        """
        try:
            params = {
                "function": "TREASURY_YIELD",
                "interval": interval,
                "maturity": maturity,
            }
            data = _make_request(params)
            return data["data"][:5]
        except Exception as e:
            return {"error": f"Error fetching treasury yield data: {str(e)}"}

    @tool
    def get_federal_funds_rate(interval: str = "monthly"):
        """
        Fetches the Federal Funds Rate for the specified interval.

        Args:
            interval (str): The interval for fetching the rate (default is 'monthly').

        Returns:
            dict: The Federal Funds Rate data.
        """
        try:
            params = {"function": "FEDERAL_FUNDS_RATE", "interval": interval}
            data = _make_request(params)
            return data["data"][:5]
        except Exception as e:
            return {"error": f"Error fetching federal funds rate data: {str(e)}"}

    @tool
    def get_consumer_price_index(interval: str = "monthly"):
        """
        Fetches the Consumer Price Index (CPI) for the specified interval.

        Args:
            interval (str): The interval for fetching CPI data (default is 'monthly').

        Returns:
            dict: The Consumer Price Index data.
        """
        try:
            params = {"function": "CPI", "interval": interval}
            data = _make_request(params)
            return data["data"][:5]
        except Exception as e:
            return {"error": f"Error fetching consumer price index data: {str(e)}"}

    @tool
    def get_inflation_rate():
        """
        Fetches the inflation rate.

        Returns:
            dict: The inflation rate data.
        """
        try:
            params = {"function": "INFLATION"}
            data = _make_request(params)
            return data["data"][:5]
        except Exception as e:
            return {"error": f"Error fetching inflation rate data: {str(e)}"}

    @tool
    def get_unemployment_rate():
        """
        Fetches the unemployment rate.

        Returns:
            dict: The unemployment rate data.
        """
        try:
            params = {"function": "UNEMPLOYMENT"}
            data = _make_request(params)
            return data["data"][:5]
        except Exception as e:
            return {"error": f"Error fetching unemployment rate data: {str(e)}"}

    def get_macroeconomic_tool(self):

        agent_executor = self.create_macroeconomic_agent()

        def analyze_macroeconomic_content(query: str, context: Optional[str] = None):
            if context:
                chain_input = {"query": f"query: {query}, context: {context}"}
            else:
                chain_input = {"query": query}
            return {
                "message": agent_executor.invoke(chain_input)["output"],
                "metadata": {"source": "macroeconomic_analyst"},
            }

        return StructuredTool.from_function(
            name="macroeconomic_analyst",
            func=analyze_macroeconomic_content,
            description="Tool for analyzing macroeconomic indicators including GDP, CPI, and unemployment rate.",
        )

    def create_macroeconomic_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Analyze the gathered macroeconomic data in great detail and provide deep insights on the economic indicators. Use suitable tools to get the data if not provided in the context.",
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(
            self.llm,
            [
                self.get_real_gdp,
                self.get_real_gdp_per_capita,
                self.get_treasury_yield,
                self.get_federal_funds_rate,
                self.get_consumer_price_index,
                self.get_inflation_rate,
                self.get_unemployment_rate,
            ],
            prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=[
                self.get_real_gdp,
                self.get_real_gdp_per_capita,
                self.get_treasury_yield,
                self.get_federal_funds_rate,
                self.get_consumer_price_index,
                self.get_inflation_rate,
                self.get_unemployment_rate,
            ],
            verbose=True,
            handle_parse_errors=True,
            max_iterations=5,
        )


class FinancialContextAnalyst:
    """
    Specialized analyst for performing detailed financial analysis on retrieved context data.
    Focuses on extracting and analyzing financial metrics from provided context.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def get_financial_context_tool(self):
        """Creates and returns a structured tool for financial context analysis."""

        def analyze_financial_context(query: str, context: str):
            """
            Analyzes financial context data based on the query requirements.

            Args:
                query: The specific analysis query
                context: Financial context data to analyze
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Answer the user query as an expert financial context analyst performing deep chain-of-thought analysis. Your role is to thoroughly analyze both the query intent and financial data through multiple analytical lenses.

1. QUERY ANALYSIS PHASE:
   - Understand query objectives:
     * Primary information being sought
     * Secondary implications
     * Required analytical depth
     * Specific metrics needed
   - Break down complex queries:
     * Core components
     * Implicit requirements
     * Analytical dependencies
     * Required calculation paths

2. CONTEXT EVALUATION PHASE:
   - Initial data assessment:
     * Available metrics and time periods
     * Data completeness and relevance
     * Quality of numerical information
     * Contextual limitations
   - Data categorization:
     * Primary metrics
     * Derived calculations
     * Temporal aspects
     * Interrelated data points

3. NUMERICAL ANALYSIS FRAMEWORK BASED ON THE QUERY AND AVAILABLE INFORMATION:
   A. Core Financial Metrics:
      - Liquidity Analysis:
        * Quick Ratio = (Cash & Equivalents + Short Term Investments + Receivables) / Current Liabilities or Quick Ratio = (Total Current Assets - Raw materials and supplies - Work in process and finish goods) / Total Current Liabilities. Use one of this formula based on the available information.  
        * Current Ratio = Current Assets / Current Liabilities
        * Working Capital = Current Assets - Current Liabilities
      
   B. Profitability Assessment:
      - Margin Analysis:
        * Gross Margin = (Revenue - COGS) / Revenue × 100
        * Operating Margin = Operating Income / Revenue × 100
        * Net Margin = Net Income / Revenue × 100
      
   C. Efficiency Metrics:
      - Asset Utilization:
        * Asset Turnover = Revenue / Average Total Assets
        * Inventory Turnover = COGS / Average Inventory
      
   D. Growth Analysis:
      - YoY Growth = (Current Value - Previous Value) / Previous Value × 100
      - CAGR = (Ending Value / Beginning Value)^(1/n) - 1

4. ANALYTICAL REASONING PROCESS:
   - For each metric/calculation:
     * Document initial value observation
     * Explain calculation methodology
     * Analyze result significance
     * Consider multiple interpretations
     * Cross-validate with related metrics
   
   - Pattern Recognition:
     * Identify trends and anomalies
     * Analyze causality chains
     * Consider seasonal factors
     * Evaluate structural changes

5. MULTI-DIMENSIONAL INSIGHTS:
   - Operational Perspective:
     * Business efficiency indicators
     * Operational bottlenecks
     * Process effectiveness
   
   - Financial Health:
     * Short-term stability
     * Long-term sustainability
     * Risk exposure levels
   
   - Strategic Implications:
     * Competitive positioning
     * Growth opportunities
     * Resource allocation efficiency

6. DATA PRESENTATION STANDARDS:
   - Numerical Formatting:
     * Maintain original precision
     * Use consistent decimal places
     * Include appropriate units
     * Show calculation steps
   
   - Table Structure:     ```markdown
     | Metric | Value | YoY Change | Analysis |
     |:-------|------:|:----------:|:---------|     ```
   
   - Visual Organization:
     * Group related metrics
     * Show temporal progression
     * Highlight key comparisons
     * Include analytical notes

CRITICAL ANALYTICAL REQUIREMENTS:

✓ THOUGHT PROCESS:
  * Show complete reasoning chain
  * Explain analytical choices
  * Document assumption basis
  * Validate conclusions

✓ NUMERICAL RIGOR:
  * Preserve exact values
  * Show all calculation steps
  * Cross-validate results
  * Maintain precision

✓ INSIGHT DEPTH:
  * Consider multiple angles
  * Analyze implications
  * Identify relationships
  * Draw supported conclusions

✓ PRESENTATION CLARITY:
  * Structured format
  * Clear progression
  * Logical flow
  * Supporting evidence

✗ ANALYTICAL CONSTRAINTS:
  * No unsupported assumptions
  * No speculative conclusions
  * No missing calculation steps
  * No unexplained methodologies

RESPONSE STRUCTURE:
1. Query Understanding
2. Data Assessment
3. Detailed Analysis
4. Supporting Calculations
5. Key Insights
6. Comprehensive Conclusions""",
                    ),
                    ("human", "{query}"),
                ]
            )

            return {
                "message": self.llm.invoke(
                    prompt.format_messages(query=query + context if context else query)
                ).content,
                "metadata": {
                    "source": "financial_context_analyst",
                    "analysis_type": "context_based",
                },
            }

        return StructuredTool.from_function(
            name="financial_context_analyst",
            func=analyze_financial_context,
            description="Specialized tool for analyzing financial data from provided context. Performs detailed ratio analysis, trend identification, and metric computation from contextual information.",
        )
