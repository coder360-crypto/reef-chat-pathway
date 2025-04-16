# Import required libraries
import json
import os
import sys
from typing import Dict

# Add parent directory to system path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import logging and concurrent processing modules
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

# Import configuration and services
from config import AgentsConfig as Config
from services.moray_services import MORAY
from langchain.callbacks import OpenAICallbackHandler
from utils.financial_generation_utils.cost_tracker import CostTracker

# Set up logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class FinancialQuestionHandler:
    """
    Handles the generation and processing of financial analysis questions for companies.
    Manages caching, parallel processing, and API interactions for financial report generation.
    """
    def __init__(self):
        self.output_dir = Path("financial_report_outputs")
        self.output_dir.mkdir(exist_ok=True)

    def get_questions(self, company: str) -> Dict[str, str]:
        """
        Args:
            company (str): Name of the company to generate questions for
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to formatted questions
        """
        return {
            "Executive Summary": f"How has {company}'s financial performance evolved over the latest fiscal year, specifically in terms of revenue growth, net income fluctuations, and cash flow trends, and what do these figures reveal about its overall operational health?",
            "Introduction": f"What overarching goals and strategic priorities does {company} aim to address in its latest 10-K report, and how do these align with its broader business objectives and stakeholder expectations?",
            "Market Overview": f"In what ways are current industry trends, technological advancements, and macroeconomic factors shaping {company}'s market dynamics, and how is it strategically positioning itself to maintain or enhance competitiveness?",
            "Investment Thesis": f"What compelling rationale does {company} present in its 10-K report to support its investment strategy, and how are potential benefits and associated risks articulated within the broader market context?",
            "Investment Analysis": f"Which financial and operational metrics highlighted in the 10-K, such as valuation ratios or profitability measures, provide insights into {company}'s market positioning and investment attractiveness, and how do these compare to industry benchmarks?",
            "Financial Analysis": f"What significant trends or anomalies can be observed in {company}'s financial statements, including revenue, operating expenses, and cash flow activities, and how do these reflect on its fiscal management and growth trajectory?",
            "Risk Management": f"What are the most critical risks identified in {company}'s 10-K, including external threats like regulatory changes or geopolitical factors, and how is the company planning to mitigate their potential impact on its operations and financial stability?",
            "Recommendations": f"What actionable recommendations are provided in the 10-K report for {company} to address identified challenges, optimize its operations, and strengthen its competitive edge, and how do these align with its stated long-term strategies?",
            "Conclusion": f"What overarching insights and forward-looking strategies does {company}'s 10-K report emphasize, and how should these be interpreted by investors and stakeholders in terms of its future market positioning and growth potential?",
        }

    def process_single_question(self, question: str, company: str) -> str:
        """
        Args:
            question (str): The question to process
            company (str): Name of the company
            
        Returns:
            str: Processed response from MORAY API
        """
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
            logger.error(f"Error processing question: {str(e)}")
            return f"Error processing question: {str(e)}"

    def load_cached_section(self, company: str, section: str) -> str:
        """Load cached section response if it exists"""
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
                        return cached_data["response"]
            return None
        except Exception as e:
            logger.error(f"Error loading cached section: {str(e)}")
            return None

    def save_section_response(self, company: str, section: str, response: str):
        """Save section response to a file"""
        try:
            company_dir = self.output_dir / company
            company_dir.mkdir(exist_ok=True)

            output_file = company_dir / f"{section.lower()}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "section": section,
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved {section} response for {company}")
        except Exception as e:
            logger.error(f"Error saving {section} response: {str(e)}")

    def process_company(self, company: str) -> Dict[str, str]:
        """
        Args:
            company (str): Name of the company to analyze
            
        Returns:
            Dict[str, str]: Dictionary containing analysis results for each section
        """
        questions = self.get_questions(company)
        results = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_section = {}

            for section, question in questions.items():
                # Try to load from cache first
                cached_result = self.load_cached_section(company, section)
                if cached_result:
                    results[section] = cached_result
                    continue

                future = executor.submit(
                    self.process_single_question, question, company
                )
                future_to_section[future] = section

            for future in as_completed(future_to_section):
                section = future_to_section[future]
                try:
                    result = future.result()
                    self.save_section_response(company, section, result)
                    results[section] = result
                except Exception as e:
                    logger.error(f"Error processing section {section}: {str(e)}")
                    results[section] = f"Error: {str(e)}"

        return results

    def get_company_analysis(self, company: str) -> Dict[str, str]:
        """Process and return analysis for a given company"""
        return self.process_company(company)


# Example usage
if __name__ == "__main__":
    # Example companies
    company = "American Express"

    processor = FinancialQuestionHandler()
    analysis = processor.get_company_analysis(company)
