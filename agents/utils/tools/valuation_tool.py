# Import standard libraries
import json
import logging
import os
import sys
from typing import Optional
from urllib.request import urlopen

# Import third-party libraries
import certifi
import requests
import yfinance as yf
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from search_tools import SerperSearchTool, TavilySearchTool
from tavily import TavilyClient

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from config import AgentsConfig as Config

logger = logging.getLogger(__name__)


class ValuationTool:
    """
    A tool for performing market analysis and company valuations.
    
    Args:
        llm: Language model instance for text generation
        max_results (int): Maximum number of search results to process
        
    Returns:
        ValuationTool: An instance of the valuation tool
    """

    def __init__(self, llm, max_results: int = 15):
        """Initialize ValuationTool with API keys from config"""
        self.max_results = max_results
        self.config = Config()
        self.llm = llm
        self.fmp_api_key = self.config.FMP_API_KEY

    def get_valuation(self, query: str) -> dict:
        """
        Returns a structured tool for market analysis and valuation insights.

        This tool provides comprehensive analysis of market opportunities by:
        - Analyzing competitor landscape and market dynamics
        - Gathering financial data and metrics for similar companies
        - Providing strategic recommendations and risk assessment
        - Evaluating business model viability and investment requirements

        The analysis includes:
        - Company data: Overview, financials, and market position of competitors
        - Strategic feedback: Market size, trends, entry barriers, and growth opportunities
        - Valuation metrics: Industry benchmarks, multiples, and capital requirements

        Returns:
            StructuredTool: A tool that performs market analysis and provides valuation insights
                          based on competitor research and financial metrics.
        """
        """Returns a StructuredTool for market analysis and valuation."""

        def run_goog_search(query: str) -> str:
            """Run the Google Serper API wrapper to search for information."""
            return SerperSearchTool().search(query)["message"]

        def tavily_search(query: str, max_results: Optional[int] = None):
            # logger.info(f"Running Tavily search with query: {query}, max_results: {max_results}")
            """Execute a Tavily search query"""
            max_results = max_results or 10
            try:
                return TavilySearchTool(
                    max_results=max_results, search_depth="advanced"
                ).search(query)["message"]
            except Exception as e:
                logger.error(f"Tavily search failed: {str(e)}")
                raise

        def extract_company_names(paragraph: str) -> list:
            logger.info("Extracting company names from text")
            """Extract company names using OpenAI API"""
            response = self.llm.invoke(
                input=[
                    {
                        "role": "user",
                        "content": f"Output Format: Company 1, Company 2, Company 3. Extract entire company names from the following text and return comma separated names only. Text:{paragraph}",
                    }
                ],
                max_tokens=100,
                temperature=0,
            )
            names = response.content
            return names

        def get_company_financials(company_symbol: str) -> dict:
            """
            Get annual revenue and growth data for a company.
            
            Args:
                company_symbol (str): Stock ticker symbol
                
            Returns:
                dict: Financial metrics including revenue, growth, ratios
            """
            try:
                company = yf.Ticker(company_symbol)
                financials = company.financials
                if financials.empty:
                    return {"error": f"No financial data found for {company_symbol}"}
                revenue = financials.loc["Total Revenue"]
                latest_revenue = revenue.iloc[0]
                prev_revenue = revenue.iloc[1]

                # Calculate YoY growth
                growth = ((latest_revenue - prev_revenue) / prev_revenue) * 100

                return {
                    "company": company_symbol,
                    "latest_annual_revenue": latest_revenue,
                    "previous_annual_revenue": prev_revenue,
                    "yoy_growth_percent": growth,
                    "market_cap": company.info.get("marketCap", None),
                    "pe_ratio": company.info.get("trailingPE", None),
                    "price_to_book": company.info.get("priceToBook", None),
                    "enterprise_value": company.info.get("enterpriseValue", None),
                    "ebitda": company.info.get("ebitda", None),
                    "profit_margins": company.info.get("profitMargins", None),
                    "debt_to_equity": company.info.get("debtToEquity", None),
                }

            except Exception as e:
                logger.error(f"Error getting data for {company_symbol}: {str(e)}")
                return {"error": f"Error getting data for {company_symbol}: {str(e)}"}

        def get_ticker(company_name: str) -> str:
            """Get ticker symbol for a company using Yahoo Finance API."""
            try:
                yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
                user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
                params = {
                    "q": company_name,
                    "quotes_count": 1,
                    "country": "United States",
                }

                res = requests.get(
                    url=yfinance, params=params, headers={"User-Agent": user_agent}
                )
                data = res.json()

                return data["quotes"][0]["symbol"]
            except Exception as e:
                # logger.error(f"Error getting ticker for {company_name}: {str(e)}")
                return None

        def get_web_query(product_idea: str) -> dict:
            """Generate optimized search query for finding relevant companies."""
            prompt = f"""As a market research expert, analyze this product idea and generate ONE precise search query.
    Product/Solution: {product_idea}

    Create a single, highly focused web search query that will find:
    - Leading companies and startups in this exact space
    - Direct competitors and similar solutions
    - Companies with comparable technology/business models

    Requirements for the query:
    1. Must be under 15 words
    2. Include specific industry terms
    3. Focus on the core business/technology aspect
    4. Exclude generic terms
    5. Use boolean operators if needed
    6. Search query should be in the form of a question, not a statement
    7. Must be in the form of a question

    Format your response as:
    QUERY: [Your single search query]"""

            try:
                response = self.llm.invoke(
                    input=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=30,
                )

                content = response.content
                query = content.split("QUERY:")[1].strip()

                return {"query": query}

            except Exception as e:
                logger.error(f"Error generating search query: {str(e)}")
                return {"query": product_idea}

        def get_dcf(ticker: str) -> dict:
            """Get Discounted Cash Flow valuation for a company ticker."""
            try:
                url = f"https://financialmodelingprep.com/api/v3/discounted-cash-flow/{ticker}?apikey={self.config.FMP_API_KEY}"
                response = urlopen(url, cafile=certifi.where())
                data = response.read().decode("utf-8")
                return json.loads(data)[0]
            except Exception as e:
                # logger.error(f"Error getting DCF for {ticker}: {str(e)}")
                return {"error": f"Failed to get DCF valuation: {str(e)}"}

        def get_rating(ticker: str) -> dict:
            """Get rating for a company ticker."""
            try:
                url = f"https://financialmodelingprep.com/api/v3/rating/{ticker}?apikey={self.config.FMP_API_KEY}"
                response = urlopen(url, cafile=certifi.where())
                data = response.read().decode("utf-8")
                return json.loads(data)[0]
            except Exception as e:
                # logger.error(f"Error getting rating for {ticker}: {str(e)}")
                return {"error": f"Failed to get rating: {str(e)}"}

        def get_company_data(query: str = None) -> dict:
            """
            Get comprehensive data about companies in the space.
            
            Args:
                query (str): Search query for finding relevant companies
                
            Returns:
                dict: Company data including overview, financials, valuations
            """
            logger.info(f"Getting company data for query: {query}")
            """Get comprehensive data about companies in the space."""
            max_results = 10

            if query is None:
                example_query = (
                    "List of fintech companies by market valuation and employee size"
                )
            else:
                example_query = get_web_query(query)["query"]
            try:
                goog_result = run_goog_search(example_query)
                tavily_result = tavily_search(example_query, max_results)
                final_search_result = goog_result + tavily_result
                companies = extract_company_names(final_search_result)
            except Exception as e:
                logger.error(f"Error running search: {str(e)}")

            merged_companies = []
            for company in companies:
                split_companies = [c.strip() for c in company.split(",")]
                merged_companies.extend(split_companies)
            companies = merged_companies[:6]

            company_data = {}
            for company in companies:
                try:
                    company_data[company] = {
                        "overview": tavily_search(
                            f"give a brief comprehensive overview of what {company} does and its valuation",
                            max_results=5,
                        )
                    }
                except Exception as e:
                    logger.error(f"Error getting overview for {company}: {str(e)}")
                    company_data[company] = {
                        "overview": f"Error getting overview: {str(e)}"
                    }

                try:
                    symbol = get_ticker(company)
                    if symbol:
                        company_data[company].update(
                            {
                                "symbol": symbol,
                                "financials": get_company_financials(symbol),
                                "dcf_valuation": get_dcf(symbol),
                                "rating": get_rating(symbol),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error getting financials for {company}: {str(e)}")
                    company_data[company].update(
                        {"error": f"Error processing company: {str(e)}"}
                    )
            return company_data

        def get_analysis(company_data: dict, product_idea: str) -> dict:
            """
            Generate comprehensive feedback on financial viability.
            
            Args:
                company_data (dict): Collected company information
                product_idea (str): Business idea to analyze
                
            Returns:
                dict: Structured analysis with valuation ranges and metrics
            """
            logger.info("Generating valuation analysis:")
            """Generate comprehensive feedback on financial viability and market positioning."""
            context = ""
            for company, data in company_data.items():
                context += f"\nCompany: {company}\n"
                if "overview" in data:
                    context += f"Overview: {data['overview']}\n"
                if "financials" in data:
                    fin = data["financials"]
                    context += f"Financial Metrics:\n"
                    context += f"- Market Cap: ${fin.get('market_cap', 'N/A')}\n"
                    context += (
                        f"- Revenue: ${fin.get('latest_annual_revenue', 'N/A')}\n"
                    )
                    context += f"- Profit Margin: {fin.get('profit_margins', 'N/A')}\n"
                    context += f"- P/E Ratio: {fin.get('pe_ratio', 'N/A')}\n"
                if "dcf_valuation" in data:
                    dcf = data["dcf_valuation"]
                    context += f"DCF Valuation:\n"
                    context += f"- DCF Value: ${dcf.get('dcf', 'N/A')}\n"
                    context += f"- Stock Price: ${dcf.get('stock_price', 'N/A')}\n"
                if "rating" in data:
                    rating = data["rating"]
                    context += f"Investment Rating:\n"
                    context += f"- Overall Rating: {rating.get('rating', 'N/A')}\n"
                    context += f"- Rating Score: {rating.get('ratingScore', 'N/A')}\n"
                    context += f"- Recommendation: {rating.get('ratingRecommendation', 'N/A')}\n"
                    context += (
                        f"- DCF Score: {rating.get('ratingDetailsDCFScore', 'N/A')}\n"
                    )
                    context += (
                        f"- ROE Score: {rating.get('ratingDetailsROEScore', 'N/A')}\n"
                    )
                    context += (
                        f"- ROA Score: {rating.get('ratingDetailsROAScore', 'N/A')}\n"
                    )
                    context += (
                        f"- DE Score: {rating.get('ratingDetailsDEScore', 'N/A')}\n"
                    )
                    context += (
                        f"- PE Score: {rating.get('ratingDetailsPEScore', 'N/A')}\n"
                    )
                    context += (
                        f"- PB Score: {rating.get('ratingDetailsPBScore', 'N/A')}\n"
                    )

            prompt = f"""You are a seasoned financial analyst and venture capitalist. Based on the following market data about companies in the space:

    {context}

    Analyze the provided market data and give a probable valuation analysis for a new company {product_idea} in this industry. Focus on:

    1. DCF Valuation Range:
    - Examine the DCF valuations of comparable companies
    - Provide an extremely wide but reasonable DCF valuation range for a new entrant, considering:
        * Industry average profit margins
        * Typical revenue growth trajectories
        * Market size and penetration rates

    2. Company Valuation:
    - Analyze key valuation metrics like:
        * P/E ratios
        * Revenue multiples
        * EBITDA multiples
        * Market cap ranges
    - Compare against industry benchmarks to determine a valuation range

    Please provide your analysis in this structured format: DO NOT INCLUDE ANY OTHER TEXT.
    PROVIDE A WIDE RANGE. IT NEED NOT BE PRECISE
    {{
        "probable_dcf_range": {{
            "lower_bound": "$X million",
            "upper_bound": "$Y million",
            "key_assumptions": {{
                "growth_rate": "",
                "profit_margins": "",
                "discount_rate": ""
            }}
        }},
        "valuation": {{
            "lower_bound": "$A million",
            "upper_bound": "$B million",
            "key_metrics": {{
                "average_pe_ratio": "",
                "average_revenue_multiple": "",
                "average_ebitda_multiple": "",
                "typical_market_cap_range": ""
            }}
        }}
    }}
    """

            try:
                response = self.llm.invoke(
                    input=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=1000,
                )

                logger.info("Successfully generated market analysis")
                return {"analysis": response.content.strip().split("\n")}
            except Exception as e:
                logger.error(f"Failed to generate feedback: {str(e)}")
                raise

        # main of get_valuation
        try:

            company_data = get_company_data(query)
            analysis = get_analysis(company_data, query)
            analysis_str = "".join(analysis["analysis"])
            analysis_str = analysis_str.replace("```json", "").replace("```", "")
            analysis_data = json.loads(analysis_str)

            # Format analysis into readable string
            analysis_content = f"""
                    Valuation Analysis:

                    1. DCF (Discounted Cash Flow) Valuation:
                    • Value Range: {analysis_data['probable_dcf_range']['lower_bound']} to {analysis_data['probable_dcf_range']['upper_bound']}
                    
                    Key Assumptions:
                    • Growth Rate: {analysis_data['probable_dcf_range']['key_assumptions']['growth_rate']}
                    • Profit Margins: {analysis_data['probable_dcf_range']['key_assumptions']['profit_margins']}
                    • Discount Rate: {analysis_data['probable_dcf_range']['key_assumptions']['discount_rate']}

                    2. Market-Based Valuation:
                    • Value Range: {analysis_data['valuation']['lower_bound']} to {analysis_data['valuation']['upper_bound']}
                    
                    Key Metrics:
                    • Industry P/E Ratio: {analysis_data['valuation']['key_metrics']['average_pe_ratio']}
                    • Revenue Multiple: {analysis_data['valuation']['key_metrics']['average_revenue_multiple']}
                    • EBITDA Multiple: {analysis_data['valuation']['key_metrics']['average_ebitda_multiple']}
                    • Market Cap Range: {analysis_data['valuation']['key_metrics']['typical_market_cap_range']}
                    """

            company_content = ""
            for company, data in company_data.items():
                company_content += f"\nCompany: {company}\n"
                if "overview" in data:
                    company_content += f"Overview: {data['overview']}\n"
                if "symbol" in data:
                    company_content += f"Symbol: {data['symbol']}\n"
                if "financials" in data and isinstance(data["financials"], dict):
                    company_content += "Financials:\n"
                    for key, value in data["financials"].items():
                        company_content += f"  {key}: {value}\n"
                if "dcf_valuation" in data and isinstance(data["dcf_valuation"], dict):
                    company_content += "DCF Valuation:\n"
                    for key, value in data["dcf_valuation"].items():
                        company_content += f"  {key}: {value}\n"
                if "rating" in data and isinstance(data["rating"], dict):
                    company_content += "Rating:\n"
                    for key, value in data["rating"].items():
                        company_content += f"  {key}: {value}\n"
            return {"message": f"{analysis_content}\n{company_content}", "metadata": {}}

        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return {"message": f"Error analyzing market: {str(e)}"}

    def get_valuation_tool(self) -> dict:
        def analyze_valuation(query: str) -> dict:
            """Analyzes market opportunity and provides valuation insights."""
            logger.info(
                f"Initializing ValuationTool with max_results={self.max_results}"
            )
            return self.get_valuation(query)

        return StructuredTool.from_function(
            name="valuation_tool",
            func=analyze_valuation,
            description="Analyzes market opportunity, competitors, and provides valuation insights for business ideas.",
            return_direct=True,
        )


if __name__ == "__main__":
    valuation_tool = ValuationTool(
        llm=ChatOpenAI(model="gpt-4o-mini")
    ).get_valuation_tool()
    response = valuation_tool.invoke(
        {"query": "suggest fintech companies by market valuation and employee size"}
    )
    print(response)
