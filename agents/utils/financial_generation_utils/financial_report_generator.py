import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict
from datetime import datetime

import markdown
from config import AgentsConfig as Config
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from utils.financial_generation_utils.financial_report_content import (
    FinancialQuestionHandler,
)
from weasyprint import HTML
from utils.financial_generation_utils.cost_tracker import CostTracker
from langchain.callbacks import OpenAICallbackHandler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    A class to generate comprehensive financial reports for companies.

    """
    def __init__(self):
        self.report_sections: Dict[str, str] = {}
        self.output_dir = Path("financial_report_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self._setup_logging()

        self.config = Config()
        self.question_processor = FinancialQuestionHandler()
        self.rectifier_prompt = """
        Review this financial report section and check for:
        - Missing critical information or metrics
        - Redundant or repeated content
        - Logical flow and consistency
        - Completeness of analysis
        
        If issues are found, resolve them while maintaining:
        - Clear, concise writing
        - Professional financial analysis
        - Proper markdown formatting
        - Consistent style throughout
        """

    def _setup_logging(self):
        self.logger = logger

    def _generate_section_content(self, section: str, context: str) -> str:
        """
        Generate content for a single section using LLM
        
        Args:
            section (str): The section name of the report
            context (str): The context information for the section
            
        Returns:
            str: Generated content for the section
        """
        try:
            prompt = self._create_prompt(section, context)
            cost_tracker = CostTracker()
            callback = OpenAICallbackHandler()
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE,
                callbacks=[callback],
            )
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            content = response.content

            # Track costs
            cost_tracker.add_usage(
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )

            return content
        except Exception as e:
            logger.error(f"Error generating section {section}: {str(e)}")
            raise

    def generate_report(self, company_name: str) -> Dict[str, str]:
        """
        Generate financial report using parallel processing
        
        Args:
            company_name (str): Name of the company to generate report for
            
        Returns:
            Dict[str, str]: Dictionary containing report message and metadata
        """
        logger.info(f"Starting report generation for {company_name}")

        # Reset cost tracker at start of report generation
        cost_tracker = CostTracker()
        cost_tracker.reset()

        try:
            # Get analysis using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=12) as executor:
                analysis_future = executor.submit(
                    self.question_processor.get_company_analysis, company_name
                )
                analysis = analysis_future.result()

            section_results = {}
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_section = {
                    executor.submit(
                        self._generate_section_content, section, context
                    ): section
                    for section, context in analysis.items()
                }

                for future in as_completed(future_to_section):
                    section = future_to_section[future]
                    try:
                        report_text = future.result()
                        section_results[section] = report_text

                        logger.info(f"Added section {section}")
                    except Exception as e:
                        logger.error(f"Error in section {section}: {str(e)}")

            self.report_sections = section_results

            combined_markdown = ""
            combined_markdown += f"# {company_name} - Financial Analysis Report\n\n"
            combined_markdown += (
                f"## Executive Summary\n\n{section_results['Executive Summary']}\n\n"
            )
            combined_markdown += (
                f"## Introduction\n\n{section_results['Introduction']}\n\n"
            )
            combined_markdown += (
                f"## Market Overview\n\n{section_results['Market Overview']}\n\n"
            )
            combined_markdown += (
                f"## Investment Thesis\n\n{section_results['Investment Thesis']}\n\n"
            )
            combined_markdown += f"## Investment Analysis\n\n{section_results['Investment Analysis']}\n\n"
            combined_markdown += (
                f"## Financial Analysis\n\n{section_results['Financial Analysis']}\n\n"
            )
            combined_markdown += (
                f"## Risk Management\n\n{section_results['Risk Management']}\n\n"
            )
            combined_markdown += (
                f"## Recommendations\n\n{section_results['Recommendations']}\n\n"
            )
            combined_markdown += f"## Conclusion\n\n{section_results['Conclusion']}\n\n"

            # Generate PDF
            output_dir = Path("files")
            output_dir.mkdir(exist_ok=True)

            css = self._get_css()
            html = markdown.markdown(combined_markdown, extensions=["tables"])
            html_with_css = f"<style>{css}</style>{html}"

            output_pdf = (
                output_dir / f"{company_name.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            HTML(string=html_with_css).write_pdf(output_pdf)

            logger.info(f"Generated PDF report at {output_pdf}")

            # Get final cost metrics
            cost_metrics = cost_tracker.get_metrics()

            # Add cost metrics to return value
            return {
                "message": "Report generated successfully",
                "metadata": {"pdf_path": output_pdf, "cost_metrics": cost_metrics},
            }

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _create_prompt(self, section: str, context: str) -> str:
        return f"""Based on the following context about {section} from the report, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report i just need the section content**

        Guidelines:
        - Do not mention the datasource name like 10-K
        - Use bullet points only for listing key metrics, findings or strategic points - do not use bullet points for the entire section - No Subpoints at all 
        - Present numerical data in well-formatted tables generate table only if required (do not generate tables is no data is available)
        - if there is no data available in context then mention not available do not make up data or write $X or X% etc
        - Write in a formal, professional tone
        - Focus on facts, figures and analysis
        - Avoid generic introductions/conclusions
        - Include all relevant data from the provided context
        - Do not generate any headings or subheadings its already present in the report i just need the section content
        - Important content and numbers should be wrapped in italics using markdown (*text*)
        - mention sources and citations of the data in the context
        
        Context:
        {context}
        Use data only from context if numbers are not present in context then mention not available 

        Format the response in markdown with:
        - Pipe tables for numerical/financial data
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
        """

    def _rectify_and_improve(self, markdown_content: str) -> str:
        """Rectify and improve the generated report content"""
        try:
            cost_tracker = CostTracker()
            callback = OpenAICallbackHandler()
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.3,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE,
                callbacks=[callback],
            )
            messages = [
                SystemMessage(content=self.rectifier_prompt),
                HumanMessage(content=markdown_content),
            ]
            response = llm.invoke(messages)

            # Track costs
            cost_tracker.add_usage(
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )

            return response.content
        except Exception as e:
            self.logger.error(f"Error in rectification phase: {str(e)}")
            return markdown_content

    def set_rectifier_prompt(self, new_prompt: str):
        """Allow customization of the rectifier prompt"""
        self.rectifier_prompt = new_prompt
        self.logger.info("Updated rectifier prompt")

    def _get_css(self) -> str:
        return """
        @page {
            @bottom-right {
                content: "Analysis Report | Page " counter(page);
                font-size: 11pt;
                font-family: "Roboto", "Open Sans", sans-serif;
                padding-top: 10px;
                color: #333333;
            }
            @bottom-line {    
                content: "";
                border-top: 2px solid black;
                margin-bottom: 10px;
                width: 100%;
            }
            margin-left: 1.5cm;
            margin-right: 1.5cm;
        }
        h1 {
            font-size: 40pt;
            color: #333333;
            font-family: "Helvetica Neue", "Arial", "Roboto", "Open Sans", sans-serif;
            font-weight: bold;
            text-align: center;
        }
        h2 {
            font-size: 24pt;
            color: #1a237e;
            border-bottom: 1px solid #1a237e;
            padding-bottom: 5px;
            margin-bottom: 0;
            font-family: "Roboto", "Open Sans", sans-serif;
            line-height: 120%;
            font-weight: bold;
            text-align: justify;
        }
        strong {
            color: #1a237e;
            font-family: "Roboto", "Open Sans", sans-serif;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 10px 0;
            font-family: "Roboto", "Open Sans", sans-serif;
            background-color: #f5f5f5;
            color: #333333;
        }
        th, td { 
            border: none;
            border-bottom: 1px solid #ddd;
            padding: 8px; 
            text-align: left;
            font-family: "Roboto", "Open Sans", sans-serif;
        }
        th { 
            background-color: #1a237e;
            color: #ffffff;
            font-family: "Roboto", "Open Sans", sans-serif;
            font-weight: bold;
        }
        td {
            color: #333333;
        }
        body {
            font-family: "Roboto", "Open Sans", sans-serif;
            color: #333333;
            text-align: justify;
        }
        em {
            font-style: italic;
            color: #00008B;
        }
        """


if __name__ == "__main__":
    import time

    start_time = time.time()
    generator = ReportGenerator()
    generator.generate_report("American Express")
    print(f"Report generation took {time.time() - start_time:.2f} seconds")
