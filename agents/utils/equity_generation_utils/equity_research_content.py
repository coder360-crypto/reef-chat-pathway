# Standard library imports
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Third-party imports
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Local imports
from config import AgentsConfig as Config
from langchain.callbacks import OpenAICallbackHandler
from langchain_openai import ChatOpenAI
from utils.equity_generation_utils.cost_tracker import CostTracker

from services.moray_services import MORAY

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def generate_equity_research_content_questions(company_name):
    """Generate initial content analysis questions for equity research report.

    Args:
        company_name (str): Name of the company being analyzed

    Returns:
        dict: Dictionary of questions organized by report section
    """
    questions = {
        # Company Overview and Performance
        "EXECUTIVE_SUMMARY": [
            {
                "question": f"""
                How has {company_name}'s financial and operational performance evolved over the latest fiscal year, specifically in terms of revenue growth, 
                net income fluctuations, and cash flow trends, and what do these figures reveal about its overall operational health? 
                Use retriever tool and search tools. Provide a comprehensive analysis of {company_name}'s financial performance and stock information, including:
                
                1) How have revenue, net income, and cash flow evolved? Include:
                   - Year-over-year (YoY) growth percentages
                   - Specific trends across quarters
                   - Key contributing factors for revenue growth or decline
                2) What does {company_name}'s balance sheet reveal about its operational health? Discuss:
                   - Debt-to-equity ratio
                   - Liquidity position (current ratio, quick ratio)
                   - Asset turnover ratios
                3) Discuss {company_name}'s stock information:
                   - Current stock price, 52-week high/low, market capitalization
                   - Historical stock price trends and performance under similar market conditions
                   - Dividend payout history and current dividend yield
                3) Who are the major shareholders, and what is the level of insider holdings? Include:
                   - Shareholding pattern (institutional vs. retail)
                   - Insider transactions over the last 12 months
                   - Float relative to market cap
                4) Evaluate key operational efficiency metrics:
                   - Customer acquisition cost (CAC) and customer lifetime value (CLV)
                   - Inventory turnover ratios compared to industry standards
                   - Supply chain efficiency metrics
                5) Compare capacity utilization and production efficiency metrics with top competitors over the past 3 years.
                
                Include all relevant numbers and quantitative data in the answer.
            """
            }
        ],
        # Market Analysis
        "MARKET_OVERVIEW": [
            {
                "question": f"""
                Analyze {company_name}'s market positioning within its industry. Address:
                
                1) Current industry trends and their implications:
                   - Key technological advancements shaping the industry
                   - Impact of macroeconomic factors (e.g., interest rates, inflation, geopolitical events)
                   - Sector-specific challenges and opportunities
                2) How is {company_name} adapting to industry trends to maintain or enhance its competitive edge?
                   - Discuss strategic investments in R&D or innovation
                   - Highlight efforts in digital transformation or automation
                3) What is the company's market share in its core segments, and how has it changed over time? Include:
                   - Data on market share trends across geographies
                   - Analysis of market penetration compared to competitors
            """
            }
        ],
        "INVESTMENT_THESIS": [
            {
                "question": f"""
                Outline the core investment thesis for {company_name}:
                
                1) What compelling rationale does {company_name} present to investors? Discuss:
                   - Strategic priorities and their alignment with market trends
                   - Unique competitive advantages or barriers to entry
                   - Growth opportunities in existing or new markets
                2) Identify potential benefits and associated risks:
                   - Discuss short-term and long-term financial outlooks
                   - Evaluate risks from regulatory, operational, or market perspectives
                3) How does {company_name}'s positioning within the industry make it an attractive investment compared to its peers?
            """
            }
        ],
        # Recent Developments
        "RECENT_NEWS_AND_TRENDS": [
            {
                "question": f"""
                Summarize the most impactful recent developments for {company_name} (past 6 months):
                
                1) Major corporate events:
                   - Acquisitions, mergers, or restructuring initiatives
                   - Launch of new products or entry into new markets
                2) Regulatory or legal updates:
                   - Changes in industry regulations affecting the company
                   - Ongoing or recent legal cases with material impact
                3) ESG and sustainability efforts:
                   - Initiatives to improve environmental, social, and governance practices
                   - Recent ESG scores or disclosures
                4) Analyst ratings and coverage changes:
                   - Summary of rating upgrades/downgrades
                   - Target price revisions and rationale
            """
            }
        ],
        # Financial and Investment Analysis
        "INVESTMENT_ANALYSIS": [
            {
                "question": f"""
                Provide a deep-dive financial analysis of {company_name}:
                
                1) Key financial metrics:
                   - Valuation ratios (P/E, EV/EBITDA, price-to-book)
                   - Profitability measures (gross margin, net margin, ROE, ROCE)
                2) Operational metrics:
                   - Production levels, sales volumes, and capacity utilization rates
                   - Efficiency measures like operating cost per unit
                3) Benchmark these metrics against industry peers:
                   - Highlight areas of outperformance or underperformance
            """
            }
        ],
        "FINANCIAL_ANALYSIS": [
            {
                "question": f"""
                Identify significant trends or anomalies in {company_name}'s financial statements:
                
                1) Revenue and expense trends:
                   - Breakdown of revenue by segment or geography
                   - Analysis of fixed vs. variable cost trends
                2) Cash flow activities:
                   - Trends in operating, investing, and financing cash flows
                   - Free cash flow generation and usage
                3) Discuss fiscal management and growth trajectory implications.
            """
            }
        ],
        "CONCALL_HIGHLIGHTS": [
            {
                "question": f"""
                Extract key takeaways from {company_name}'s latest earnings call:
                
                1) Updates on operational performance:
                   - Production targets and capacity utilization rates
                   - Status of major projects or investments
                2) Management commentary on:
                   - Market conditions and competitive landscape
                   - Strategic priorities for the next fiscal year
                3) Notable discussions during Q&A:
                   - Responses to analyst questions
                   - Clarity on potential risks or opportunities
            """
            }
        ],
        "INVESTMENT_RATIONALE": [
            {
                "question": f"""
                Use retriever tool and search tools. Provide a comprehensive analysis of {company_name} including:
                
                1) Core business model and revenue drivers
                2) Key valuation metrics and their implications
                3) Comparison with top 3 competitors:
                  - Profitability, efficiency, and liquidity metrics
                  - Working capital and capex patterns
                
                Include all relevant numbers, metrics and quantitative data in the answer.
            """
            }
        ],
    }
    return questions


def generate_equity_research_final_questions(company_name):
    """Generate final analysis questions for equity research report.

    Args:
        company_name (str): Name of the company being analyzed

    Returns:
        dict: Dictionary of questions organized by report section
    """
    questions = {
        # Investment Recommendation
        "RATING_AND_TARGET_PRICE": [
            {
                "question": f"""
                What is the recommended investment action for {company_name} (BUY, SELL, HOLD), and why is the security deemed mispriced? 
                Provide a detailed valuation analysis that includes:
                
                1) Target price and the methodology used to arrive at it (e.g., DCF, SOTP, comparable multiples)
                2) Key factors influencing the recommendation, such as earnings growth, market share, or competitive positioning
                3) A comparison of {company_name}'s current valuation metrics (e.g., P/E, EV/EBITDA, P/B) with industry peers
                4) Discussion on whether the market has underpriced or overpriced the stock and the reasons behind it
                5) Sensitivity analysis: How does the target price change under optimistic, baseline, and pessimistic scenarios?
                
                Ensure all metrics and figures are supported with clear calculations and data.
            """
            }
        ],
        # Risk Assessment
        "RISK_ANALYSIS": [
            {
                "question": f"""
                What are the key risks associated with {company_name}'s business operations and financial stability? Provide a 
                comprehensive analysis that includes:
                
                1) Internal risks such as operational inefficiencies, cost overruns, or execution delays
                2) External risks including:
                   - Regulatory changes, legal challenges, or policy uncertainties
                   - Geopolitical factors affecting supply chain or market access
                   - Commodity price volatility or macroeconomic factors
                3) Potential impact of each risk on financial performance, stock valuation, and operational stability
                4) Mitigation strategies and how effective they are in reducing exposure to these risks
                5) Industry-specific risks (e.g., environmental or technological disruption) and their long-term implications
                
                Provide quantified impacts where possible and prioritize risks based on their severity and likelihood.
            """
            }
        ],
        # Strategic Recommendations
        "RECOMMENDATIONS": [
            {
                "question": f"""
                What actionable recommendations and key takeaways are provided in {company_name}'s reports and latest earnings call? 
                Focus on:
                
                1) Management's response to identified challenges and opportunities
                2) Strategies to optimize operations and improve profitability
                3) Key initiatives for growth, such as product launches, market expansions, or R&D investments
                4) Forward guidance including quantitative targets for revenue, margins, and CAPEX
                5) Key insights from management's responses to analyst questions during Q&A sessions
                6) ESG and sustainability strategies that align with long-term goals
                
                Summarize these recommendations and initiatives with specific metrics, timelines, and anticipated outcomes.
            """
            }
        ],
        # Forward-Looking Analysis
        "CONCLUSION": [
            {
                "question": f"""
                What overarching insights and forward-looking strategies are emphasized in {company_name}'s reports? Address:
                
                1) Key takeaways about the company's market positioning, strategic focus, and growth trajectory
                2) Predictions on future industry trends and how {company_name} is positioned to capitalize on them
                3) Summary of growth drivers, including geographic expansion, innovation, or operational efficiency
                4) Long-term financial outlook and implications for investors
                5) Recommendations for stakeholders on monitoring critical factors that may influence {company_name}'s performance
                
                Provide a forward-looking narrative that connects the company's current position to its future potential.
            """
            }
        ],
        # Financial Data Tables
        "APPENDIX": [
            {
                "question": f"""
                Generate a comprehensive set of financial data tables for {company_name}. Include:
                
                1) Income Statement:
                   - Revenue, gross profit, EBITDA, EBIT, PBT, PAT, and EPS for the last 3 fiscal years
                   - YoY growth rates and margin analysis
                2) Balance Sheet:
                   - Share capital, reserves, net worth, total assets, liabilities, debt levels, and working capital
                   - Leverage ratios and trends over the past 3 years
                3) Cash Flow Statement:
                   - Operating, investing, and financing cash flows
                   - Capital expenditures, free cash flow (FCF), and dividend payouts
                4) Key Financial Ratios:
                   - Profitability ratios (ROE, ROA, ROCE, EBITDA margin)
                   - Valuation metrics (P/E, P/B, EV/EBITDA)
                   - Liquidity and leverage ratios (current ratio, debt-to-equity)
                5) Segment-wise Revenue Analysis:
                   - Revenue breakdown by business segments and geographies (if available)
                   - Comparison of segment performance across fiscal years
                6) Quarterly Trends:
                   - Financial performance by quarter, highlighting seasonality or cyclical trends
                
                Format the tables with clear labels and units (e.g., billions, millions). Highlight key trends or anomalies where applicable.
            """
            }
        ],
    }
    return questions


class EquityResearchQuestionProcessor:
    """A class to process and analyze equity research questions.
    
    This class handles the generation, processing, and caching of equity research
    questions and their responses for company analysis.

    Args:
        None

    Attributes:
        content_analysis (Dict): Stores content analysis results
        output_dir (Path): Directory for storing cached outputs
    """
    def __init__(self):
        config = Config()
        self.content_analysis = {}
        self.output_dir = Path("equity_research_outputs")
        self.output_dir.mkdir(exist_ok=True)

    def load_cached_section(self, company: str, section: str) -> List[str]:
        """Load cached section responses if they exist"""
        try:
            company_dir = self.output_dir / company
            cache_file = company_dir / f"{section.lower()}.json"
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    # Check if cache is less than 24 hours old
                    cache_time = datetime.fromisoformat(cached_data["timestamp"])
                    if datetime.now() - cache_time < timedelta(hours=24):
                        logger.info(f"Using cached data for {section} in {company}")
                        return cached_data["responses"]
            return None
        except Exception as e:
            logger.error(f"Error loading cached section: {str(e)}")
            return None

    def save_section_response(self, company: str, section: str, responses: List[str]):
        """Save section responses to a file"""
        try:
            # Create company directory
            company_dir = self.output_dir / company
            company_dir.mkdir(exist_ok=True)

            # Save responses to file
            output_file = company_dir / f"{section.lower()}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "section": section,
                        "responses": responses,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved {section} responses for {company}")
        except Exception as e:
            logger.error(f"Error saving {section} responses: {str(e)}")

    def process_single_content_question(self, question: str, company: str) -> str:
        """Process a single question and return its result"""
        try:
            config = Config()
            cost_tracker = CostTracker()

            logger.info(f"[CONTENT Q] Starting to process question for {company}")
            logger.info(f"[CONTENT Q] Question text: {question[:200]}...")

            process_query = MORAY()
            result = process_query.process_user_query(
                query=question,
                context=[f""],
                llm="gpt-4o",
                jwt_token=config.JWT_TOKEN,
            )

            # Track costs from the result metrics
            cost_tracker.add_usage(
                input_tokens=result["execution_metrics"]["total_metrics"][
                    "input_tokens"
                ],
                output_tokens=result["execution_metrics"]["total_metrics"][
                    "output_tokens"
                ],
                cost=result["execution_metrics"]["total_metrics"]["total_cost"],
            )

            return result["results"][-1]["join"]["messages"][-1].content

        except Exception as e:
            logger.error(
                f"[CONTENT Q] Error processing question: {str(e)}", exc_info=True
            )
            return f"Error processing question: {str(e)}"

    def process_single_final_question(self, question: str, company: str) -> str:
        """Process a single question and return its result"""
        try:
            config = Config()
            logger.info(
                f"Processing single final question for {company}: {question[:100]}..."
            )
            cost_tracker = CostTracker()
            process_query = MORAY()
            result = process_query.process_user_query(
                query=question,
                context=[f""],
                llm="gpt-4o",
                jwt_token=config.JWT_TOKEN,
            )
            # Track costs from the result metrics
            cost_tracker.add_usage(
                input_tokens=result["execution_metrics"]["total_metrics"][
                    "input_tokens"
                ],
                output_tokens=result["execution_metrics"]["total_metrics"][
                    "output_tokens"
                ],
                cost=result["execution_metrics"]["total_metrics"]["total_cost"],
            )

            return result["results"][-1]["join"]["messages"][-1].content

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            return f"Error processing question: {str(e)}"

    def process_content_questions(self, company: str) -> Dict[str, List[str]]:
        """Process content questions for a company in parallel"""
        logger.info(f"[PROCESS] Starting to process content questions for {company}")

        try:
            questions = generate_equity_research_content_questions(company)
            results = {}

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_section = {}

                for section, question_list in questions.items():
                    # Try to load from cache first
                    cached_results = self.load_cached_section(company, section)
                    if cached_results:
                        results[section] = cached_results
                        continue

                    logger.info(f"[PROCESS] Processing section: {section}")
                    # Submit section processing to thread pool
                    future = executor.submit(
                        self._process_section, company, section, question_list
                    )
                    future_to_section[future] = section

                # Collect results as they complete
                for future in as_completed(future_to_section):
                    section = future_to_section[future]
                    try:
                        section_results = future.result()
                        self.save_section_response(company, section, section_results)
                        results[section] = section_results
                    except Exception as e:
                        logger.error(f"Error processing section {section}: {str(e)}")
                        results[section] = [f"Error: {str(e)}"]

            self.content_analysis = results
            return results

        except Exception as e:
            logger.error(
                f"[PROCESS] Error in process_content_questions: {str(e)}", exc_info=True
            )
            raise

    def _process_section(
        self, company: str, section: str, question_list: List[Dict]
    ) -> List[str]:
        """Process all questions for a single section"""
        section_results = []
        for q in question_list:
            answer = self.process_single_content_question(q["question"], company)
            section_results.append(answer)
        return section_results

    def process_final_questions(self, company: str) -> Dict[str, List[str]]:
        """Process final questions for a company in parallel"""
        logger.info(f"Processing final questions for {company}")

        questions = generate_equity_research_final_questions(company)
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_section = {}

            for section, question_list in questions.items():
                # Try to load from cache first
                cached_results = self.load_cached_section(company, section)
                if cached_results:
                    results[section] = cached_results
                    continue

                future = executor.submit(
                    self._process_section, company, section, question_list
                )
                future_to_section[future] = section

            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    section_results = future.result()
                    self.save_section_response(company, section, section_results)
                    results[section] = section_results
                except Exception as e:
                    logger.error(f"Error processing section {section}: {str(e)}")
                    results[section] = [f"Error: {str(e)}"]

        self.final_analysis = results
        return results

    def get_content_analysis(self, company: str) -> Dict[str, List[str]]:
        """Get content analysis for a company"""
        logger.info("Starting get_content_analysis for %s", company)
        try:
            return self.process_content_questions(company)
        except Exception as e:
            logger.error(f"Error in get_content_analysis: {str(e)}", exc_info=True)
            return {}

    def get_final_analysis(self, company: str) -> Dict[str, List[str]]:
        """Get final analysis for a company"""
        logger.info("Starting get_final_analysis for %s", company)
        try:
            return self.process_final_questions(company)
        except Exception as e:
            logger.error(f"Error in get_final_analysis: {str(e)}", exc_info=True)
            return {}

    def extract_recommendation(
        self, final_analysis: Dict[str, List[str]]
    ) -> Tuple[str, List[str]]:
        """Extract recommendation and target risks from final analysis"""
        config = Config()
        rating_results = final_analysis.get("RATING_AND_TARGET_PRICE", [])

        recommendation = ""
        cost_tracker = CostTracker()
        callback = OpenAICallbackHandler()

        try:
            if rating_results:

                # Create LLM instance for filtering
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    api_key=Config().OPENAI_API_KEY,
                    callbacks=[callback],
                )

                # Process rating results
                messages = [
                    {
                        "role": "system",
                        "content": "You are a financial analyst assistant. Convert the given analysis into a single string BUY,SELL,HOLD",
                    },
                    {
                        "role": "user",
                        "content": f"Given the following analysis: {rating_results[0]}, give me a single string BUY,SELL,HOLD as recommendation",
                    },
                ]
                rating_response = llm.invoke(messages).content

                # Track costs after the call
                cost_tracker.add_usage(
                    input_tokens=callback.prompt_tokens,
                    output_tokens=callback.completion_tokens,
                    cost=callback.total_cost,
                )

                recommendation = rating_response

        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")

        return recommendation

def main():
    # Create an instance of the processor
    processor = EquityResearchQuestionProcessor()
    # Test with a sample company
    company_name = "Cipla"

    print(f"Processing equity research questions for {company_name}...")

    # Get final analysis
    final_analysis = processor.get_final_analysis(company_name)

    # Extract recommendation
    recommendation = processor.extract_recommendation(final_analysis)
    print(f"Recommendation: {recommendation}")

if __name__ == "__main__":
    main()

# add concall key takeaways
