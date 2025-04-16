# Import system modules
import os
import sys

# Add parent directory to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import required libraries
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import third-party libraries
import fitz
import markdown
import pytesseract
from config import AgentsConfig
from json_repair import repair_json
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from PIL import Image
from pydantic import BaseModel, Field
from tavily import TavilyClient
from utils.tools.graph_tool import GraphTool
from weasyprint import HTML

# Configure logging settings
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = AgentsConfig()


class PDFParser:
    def __init__(self):
        pass

    def get_pdf_text(self, file_path):
        """
        Extracts text from a PDF file using OCR.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        text = ""
        try:
            document = fitz.open(file_path)
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        except Exception as e:
            print(f"An error occurred while extracting text from the PDF: {e}")
        return text


class ReportGenerator:
    def __init__(self, charts: list[str]):
        self.chart_files = charts

    def convert_to_markdown(self, esg_data, level=1, chart_state=None):
        """Recursively converts ESG data to markdown with appropriate heading levels"""
        # Initialize chart state on first call
        if chart_state is None:
            chart_state = {"index": 0}

        report = ""

        if isinstance(esg_data, dict):
            for key, value in esg_data.items():
                heading = "#" * level
                formatted_key = key.replace("_", " ").capitalize()
                report += f"{heading} {formatted_key}\n"

                if isinstance(value, dict) and "Common_Metrics_Comparison" in value:
                    # Handle Common Metrics Comparison section
                    metrics_comparison = value["Common_Metrics_Comparison"]

                    # Process each metric
                    for metric_key, metric_value in metrics_comparison.items():
                        report += f"### {metric_key.replace('_', ' ').capitalize()}\n\n"

                        # Create markdown table
                        if isinstance(metric_value, dict):
                            report += "| Company | Value |\n"
                            report += "|---------|--------|\n"
                            for company, data in metric_value.items():
                                if company != "Gap_Analysis":
                                    report += f"| {company} | {data} |\n"
                            report += "\n"

                            if "Gap_Analysis" in metric_value:
                                report += f"**Gap Analysis:** {metric_value['Gap_Analysis']}\n\n"

                            # Add corresponding chart if available
                            if chart_state["index"] < len(self.chart_files):
                                chart_file = self.chart_files[chart_state["index"]]
                                abs_path = os.path.abspath(chart_file)
                                report += (
                                    f"\n![{metric_key} Comparison]({abs_path})\n\n"
                                )
                                chart_state["index"] += 1  # Move to next chart
                else:
                    if isinstance(value, (dict, list)):
                        report += self.convert_to_markdown(
                            value, level + 1, chart_state
                        )
                    else:
                        report += f"{value}\n\n"

        elif isinstance(esg_data, list):
            for item in esg_data:
                report += f"- {item}\n"
            report += "\n"

        return report

    def generate_esg_report(self, company_name: str, esg_data: dict) -> str:
        """Generate HTML template for ESG report."""
        try:
            logger.info(f"Starting ESG report generation for {company_name}")

            logger.debug("Converting JSON data to markdown sections")
            markdown_content = self.convert_to_markdown(esg_data)

            # Modify this line to allow raw HTML
            html_content = markdown.markdown(
                markdown_content, extensions=["markdown.extensions.tables"]
            )

            html_template = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>ESG Report - {company_name}</title>
            <style>
            /* Basic styles */
            body {{
            font-family: Arial, sans-serif;
            line-height: 1.4;
            padding: 15px;
            color: #333;
            font-size: 11pt;
            }}

            /* Heading styles */
            h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 8px;
            margin-bottom: 20px;
            font-size: 18pt;
            text-transform: uppercase;
            }}
            h2 {{
            color: #34495e;
            border-left: 3px solid #34495e;
            padding-left: 8px;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 16pt;
            text-transform: uppercase;
            }}

            h3 {{
            color: #2980b9;
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 14pt;
            text-transform: uppercase;
            }}

            h4 {{
            color: #3498db;
            margin-top: 15px;
            margin-bottom: 8px;
            font-size: 12pt;
            text-transform: uppercase;
            }}

            /* Table styles */
            table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 10pt;
            }}

            th {{
            background: #2c3e50;
            color: white;
            font-weight: bold;
            padding: 8px;
            text-align: left;
            }}

            td {{
            padding: 6px 8px;
            border-bottom: 1px solid #ddd;
            }}

            tr:nth-child(even) {{
            background: #f8f9fa;
            }}

            /* Gap Analysis styles */
            strong {{
            display: block;
            margin-top: 10px;
            margin-bottom: 10px;
            font-size: 10pt;
            }}

            /* Image styles */
            img {{
            max-width: 90%;
            height: auto;
            margin: 10px auto;
            display: block;
            border-radius: 3px;
            }}

            /* Section styles */
            .section {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            }}

            /* List styles */
            ul {{
            margin: 8px 0;
            padding-left: 15px;
            }}

            li {{
            margin-bottom: 5px;
            font-size: 10pt;
            }}

            p {{
            margin: 8px 0;
            font-size: 10pt;
            }}
            </style>
            </head>
            <body>
            {html_content}
            </body>
            </html>
            """

            return html_template

        except Exception as e:
            logger.error(f"Error in generate_esg_report: {str(e)}")
            raise

    def generate_output_path(self, company_name: str, html_content: str):
        """Generate PDF from HTML content."""
        try:
            logger.info(f"Starting PDF generation for {company_name}")

            # Create output directory
            os.makedirs("files", exist_ok=True)
            logger.info("Successfully created output directory")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                f'files/{company_name.replace(" ", "_")}_ESG_report_{timestamp}.pdf'
            )
            logger.info(f"Output path will be: {output_path}")

            # Create WeasyPrint HTML object
            html_obj = HTML(string=html_content, base_url="")
            # Write PDF
            try:
                html_obj.write_pdf(target=output_path)
                logger.info("Successfully wrote PDF")
            except Exception as e:
                logger.info(f"Error writing PDF: {str(e)}")

            logger.info(f"Successfully generated PDF at {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error in PDF generation: {str(e)}")
            raise

    def generate_pdf(self, company_name: str, esg_data: dict):
        """Main function to generate ESG report."""
        try:
            logger.info(f"Starting main process for {company_name}")

            # Validate inputs
            if not company_name:
                raise ValueError("Company name is empty")
            if not esg_data:
                raise ValueError("ESG data is empty")

            html_content = self.generate_esg_report(company_name, esg_data)
            logger.debug("Successfully generated HTML content")

            pdf_path = self.generate_output_path(company_name, html_content)
            logger.debug("Successfully generated PDF")

            logger.info(f"ESG report generation completed successfully at {pdf_path}")
            return pdf_path
        except Exception as e:
            logger.error(f"Error in main function: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise


class TextToJSONConverter:
    """Class to convert text data into structured JSON using LLM."""

    def __init__(self, llm: BaseChatModel):
        """Initialize with LLM."""
        self.llm = llm

    def convert_to_json(self, text: str) -> list[dict]:
        """
        Convert text data to JSON format using LLM.

        Args:
            text (str): Input text to convert
            schema_description (str, optional): Description of expected JSON schema

        Returns:
            dict: Structured JSON data
        """
        try:
            logger.info("Starting text to JSON conversion")

            # Build prompt
            prompt = f"""Task: Convert the following plain-text document into a structured JSON format. Ensure that the output matches the structure of the provided example JSON. Retain all the information and categorize the data properly.
            Example JSON:
            [
            {{
            "Company": "Company Name"
            }},
            {{
            "Category1": {{
            "Subcategory1": {{
            "unit": "Unit",
            "year": Value
            }},
            "Subcategory2": {{
            "unit": "Unit",
            "year": Value
            }}
            }},
            "Category2": {{
            "Subcategory1": {{
            "unit": "Unit",
            "year": Value
            }}
            }}
            }}
            ]
            Guidelines:

            1. Extract all numerical data, units, and categories as specified in the text.
            2. Use nested JSON objects for categories, subcategories, and their attributes.
            3. Use appropriate keys and structure them according to the JSON format provided in the example.
            4. Ensure accuracy in units (e.g., %, Persons, Years, Age, etc.) and their respective values.
            
            Text:
            {text}
            """

            # Get LLM response
            response = self.llm.invoke(
                [{"role": "system", "content": prompt, "temperature": 0.0}]
            )

            # Parse and validate JSON
            json_data = json.loads(repair_json(response.content))
            logger.info("Successfully converted text to JSON")

            return json_data

        except Exception as e:
            logger.error(f"Error converting text to JSON: {str(e)}")
            raise


# Base schema for all comparison tools
class TargetCompanySchema(BaseModel):
    """Schema for target company tool."""

    query: str = Field(..., description="Queries to search for")


class TargetCompanyFinderTool(BaseTool):
    """Tool for identifying target company from the query."""

    name: str = "target_company_finder_tool"
    description: str = "Identify target company from the query"
    args_schema: type[BaseModel] = TargetCompanySchema
    llm: BaseChatModel

    def _run(self, query: str) -> dict:
        prompt = f"""Identify the target company from the query which needs to be compared \
            with its competitors. Output only one company. Output JSON format: {{'target_company': 'target_company_name'}}
        Query: {query}"""
        """Execute the search."""
        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )
        return json.loads(repair_json(response.content))["target_company"]

    def get_target_company(self, query: str) -> str:
        """Identify target company from the query."""
        return self._run(query)


class SearchSchema(BaseModel):
    """Schema for search tool."""

    query: list[str] = Field(..., description="Queries to search for")


class SearchTool(BaseTool):
    """Tool for searching for information."""

    name: str = "search_tool"
    description: str = "Search for information using a query"
    args_schema: type[BaseModel] = SearchSchema
    llm: BaseChatModel

    def _extract_top_search_results(
        self, search_response: dict, num_results: int = 1
    ) -> list:
        """Extract top N search results by score."""
        if not isinstance(search_response, dict):
            search_response = json.loads(search_response)

        # Sort results by score and get top N
        sorted_results = sorted(
            search_response.get("results", []),
            key=lambda x: x.get("score", 0),
            reverse=True,
        )[:num_results]

        return [result["content"] for result in sorted_results]

    def _process_search_result(
        self, query: str, client: TavilyClient
    ) -> tuple[str, str]:
        """Process a single search query and return results."""
        try:
            result = client.search(query)
            logger.info(f"Successfully executed search for query: {query[:50]}...")

            # Process the result with LLM
            prompt = f"""Your task is to process the search results and return response which is completely in third person and ensure all \
                the Web sources or citations are removed. Do not include any other text than the results. Do not summarize the analysis and \
                ensure no information is lost.
            {result}"""

            response = self.llm.invoke(
                [{"role": "system", "content": prompt, "temperature": 0.0}]
            )

            return query, response.content

        except Exception as e:
            logger.error(f"Error in search for query {query[:50]}...: {str(e)}")
            return query, f"Error: {str(e)}"

    def _run(self, queries: list[str]) -> dict:
        """Execute the search."""
        try:
            search_results = {}
            client = TavilyClient(api_key=config.TAVILY_API_KEY)

            if isinstance(queries, list):
                # Process queries in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(queries)) as executor:
                    # Submit all search tasks
                    future_to_query = {
                        executor.submit(
                            self._process_search_result, query, client
                        ): f"query_{i+1}"
                        for i, query in enumerate(queries)
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_query):
                        query_id = future_to_query[future]
                        try:
                            _, result = future.result()
                            search_results[query_id] = result
                        except Exception as e:
                            logger.error(f"Error processing {query_id}: {str(e)}")
                            search_results[query_id] = f"Error: {str(e)}"

            return search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return json.dumps({"error": str(e)})

    def get_company_search_queries(self, company_name: str) -> list[str]:
        """Generate specific search queries for different aspects of company ESG analysis."""

        queries = [
            # Company Overview Query
            f"{company_name} company overview business activities revenue products services",
            # Industry Classification Query
            f"{company_name} industry classification main competitors market share",
            # Company ESG Initiatives Query
            f"{company_name} recent ESG initiatives sustainability programs controversies 2023 2024",
        ]

        return queries


class ComparisonSchema(BaseModel):
    """Base schema for ESG comparison tools."""

    target_company_data: str = Field(
        ..., description="JSON string containing target company data"
    )
    comparison_companies_data: str = Field(
        ..., description="JSON string containing data of companies to compare against"
    )
    context: Optional[str] = Field(
        None, description="Additional context or specific metrics to focus on"
    )


class AggregatorSchema(BaseModel):
    """Schema for ESG aggregator tool."""

    env_analysis: dict = Field(..., description="Results from environmental analysis")
    social_analysis: dict = Field(..., description="Results from social analysis")
    gov_analysis: dict = Field(..., description="Results from governance analysis")
    target_company_name: str = Field(
        ..., description="Name of the target company being analyzed"
    )


class EnvironmentComparisonTool(BaseTool):
    """Tool for comparing environmental metrics between companies."""

    name: str = "environment_comparison_tool"
    description: str = """
    Compare and analyze Environmental metrics between companies.
    Use this tool to analyze:
    1. Carbon emissions and energy usage
    2. Resource management
    3. Environmental initiatives
    4. Sustainability practices
    """
    args_schema: type[BaseModel] = ComparisonSchema
    llm: BaseChatModel

    def _run(
        self, target_data: str, comparison_data: str, context: Optional[str] = None
    ) -> dict:
        prompt = f"""Instructions:
        You are tasked with analyzing and comparing the Environmental performance of the target company against multiple competitors using the provided Environmental data. The goal is to conduct a comprehensive analysis that automatically identifies common metrics between the target company and comparison companies, and provides both quantitative comparisons and qualitative insights on those metrics. If some metrics are unique to certain companies, use those as additional insights to enhance the report.
        <target_company>
        {target_data}
        </target_company>

        <comparison_companies>
        {comparison_data}
        </comparison_companies>

        {f'Additional context: {context}' if context else ''}

        Key Analysis Requirements:
        1. Identify Common Environmental Metrics:
        Automatically identify the common metrics across the target company and the comparison companies (e.g., carbon emissions, energy usage, water consumption, etc.).
        2. Compare these metrics quantitatively (e.g., numerical values such as tons of CO2, kWh, recycling rate, etc.).
        3. Analyze and Compare:
        - Perform quantitative comparisons of common metrics between the target company and competitors.
        - For metrics unique to the target company or comparison companies, provide qualitative insights that help explain the differences and how they contribute to the overall ESG performance.
        4. Provide Key Insights:
        Focus on how the target company stands out in specific Environmental metrics compared to peers.
        Where unique data exists for any company, analyze how these unique initiatives or data points may affect the company’s overall Environmental performance or industry position.
        5. Gap Identification:
        - Metrics where the target company lags behind.
        - Metrics where the target company leads.
        - Areas where the target company could improve or invest more to align with industry standards.
        6. Identify areas where the target company leads or lags behind the competition.

        Create an intermediate analysis in this format:
        {{
        "Target_Company_Analysis": {{
        "Strengths": [
        "Areas where the target company leads in specific Environmental metrics."
        ],
        "Weaknesses": [
        "Areas needing improvement or metrics where the target company lags behind competitors."
        ],
        "Competitive_Position": "Overall position of the target company compared to industry peers, including both strengths and gaps in Environmental performance.",
        "Unique_Initiatives": [
        "Distinctive initiatives or programs that set the target company apart in terms of environmental efforts."
        ]
        }},
        "Comparative_Analysis":{{
        "Common_Metrics_Comparison":
        {{
        "Metric_1": {{
        "Target_Company": "Numerical value or description of performance",
        "Competitor_1": "Comparison value from competitor",
        "Competitor_2": "Comparison value from competitor",
        "Gap_Analysis": "Analysis of performance gaps between the target company and competitors for this specific metric."
        }},
        "Metric_2": {{
        "Target_Company": "Numerical value or description of performance",
        "Competitor_1": "Comparison value from competitor",
        "Competitor_2": "Comparison value from competitor",
        "Gap_Analysis": "Analysis of performance gaps between the target company and competitors for this specific metric."
        }}
        }},
        "Industry_Leaders": [
        "Companies leading in specific environmental metrics based on available data."
        ],
        "Best_Practices": [
        "Best practices in the industry that the target company can emulate to improve its Environmental performance."
        ],
        "Gap_Analysis": "In-depth analysis of the gaps between the target company and the leading companies in the Environmental space. This includes areas where the target company excels and areas where it needs to catch up."
        }},
        "Recommendations": {{
        "Short_Term": [
        "Immediate actions for improving performance in specific Environmental metrics where the target company lags."
        ],
        "Long_Term": [
        "Strategic initiatives and long-term goals that the target company should pursue to improve its Environmental standing."
        ],
        "Best_Practice_Adoption": [
        "Recommendations on specific best practices from leading companies that the target company should consider implementing to enhance its Environmental initiatives."
        ]
        }}
        }}
        Replace Target_Company with the name of the target company.
        Replace Competitor_1 and Competitor_2 with the names of the comparison companies according to the data you have collected.
        """

        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )
        return json.loads(repair_json(response.content))


class SocialComparisonTool(BaseTool):
    """Tool for comparing social metrics between companies."""

    name: str = "social_comparison_tool"
    description: str = """
    Compare and analyze Social metrics between companies.
    Use this tool to analyze:
    1. Employee welfare and diversity
    2. Community engagement
    3. Human rights practices
    4. Labor conditions
    """
    args_schema: type[BaseModel] = ComparisonSchema
    llm: BaseChatModel

    def _run(
        self, target_data: str, comparison_data: str, context: Optional[str] = None
    ) -> dict:
        prompt = f"""You are tasked with analyzing and comparing the social performance of the target company against multiple competitors using the provided ESG data. The goal is to conduct a comprehensive analysis that automatically identifies common social metrics between the target company and comparison companies, and provides both quantitative comparisons and qualitative insights on those metrics. If some metrics are unique to certain companies, use those as additional insights to enhance the report.

        <target_company>
        {target_data}
        </target_company>

        <comparison_companies>
        {comparison_data}
        </comparison_companies>

        {f'Additional context: {context}' if context else ''}
        Key Analysis Requirements:
        1. Identify Common Social Metrics:
        Automatically identify the common social metrics across the target company and the comparison companies (e.g., employee diversity, worker welfare, community engagement, etc.).
        2. Compare these metrics quantitatively:
        Compare numerical values of common social metrics (e.g., diversity percentages, turnover rates, safety incident rates, etc.).
        3. Analyze and Compare:
        - Perform quantitative comparisons of common social metrics between the target company and competitors.
        - For metrics unique to the target company or comparison companies, provide qualitative insights to explain how these unique factors contribute to the overall social performance.
        4. Provide Key Insights:
        - Focus on how the target company stands out in specific social metrics compared to peers.
        - Where unique data exists for any company, analyze how these unique initiatives or data points may impact the company’s social performance or industry position.
        5. Gap Identification:
        - Identify social metrics where the target company lags behind.
        - Identify social metrics where the target company leads.
        - Identify areas where the target company could improve or invest more to align with industry standards which will help in improving its social performance.
        6. Identify Areas Where the Target Company Leads or Lags Behind:
        - Identify where the target company excels or has the potential to improve relative to competitors in key social metrics.
        Create an intermediate analysis in this format:
        {{
        "Target_Company_Analysis": {{
        "Strengths": [
        "Areas where the target company leads in specific social metrics."
        ],
        "Weaknesses": [
        "Areas needing improvement or metrics where the target company lags behind competitors."
        ],
        "Competitive_Position": "Overall position of the target company compared to industry peers, including both strengths and gaps in social performance.",
        "Unique_Initiatives": [
        "Distinctive social initiatives or programs that set the target company apart in terms of employee welfare, diversity, or community impact."
        ]
        }},
        "Comparative_Analysis": {{
        "Common_Metrics_Comparison": {{
        "Metric_1": {{
        "Target_Company": "Numerical value or description of social performance",
        "Competitor_1": "Comparison value from competitor",
        "Competitor_2": "Comparison value from competitor",
        "Gap_Analysis": "Analysis of performance gaps between the target company and competitors for this specific metric."
        }},
        "Metric_2": {{
        "Target_Company": "Numerical value or description of social performance",
        "Competitor_1": "Comparison value from competitor",
        "Competitor_2": "Comparison value from competitor",
        "Gap_Analysis": "Analysis of performance gaps between the target company and competitors for this specific metric."
        }}
        }},
        "Industry_Leaders": [
        "Companies leading in specific social metrics, based on available data."
        ],
        "Best_Practices": [
        "Best practices in the industry that the target company can emulate to improve its social performance."
        ],
        "Gap_Analysis": "In-depth analysis of the gaps between the target company and the leading companies in the social impact space. This includes areas where the target company excels and areas where it needs to catch up.",
        }},
        "Recommendations": {{
        "Short_Term": [
        "Immediate actions for improving performance in specific social metrics where the target company lags."
        ],
        "Long_Term": [
        "Strategic social initiatives and long-term goals that the target company should pursue to improve its social impact."
        ],
        "Best_Practice_Adoption": [
        "Recommendations on specific best practices from leading companies that the target company should consider implementing to enhance its social programs."
        ]
        }}
        }}
        Replace Target_Company with the name of the target company.
        Replace Competitor_1 and Competitor_2 with the names of the comparison companies according to the data you have collected.
        """

        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )
        return json.loads(repair_json(response.content))


class GovernanceComparisonTool(BaseTool):
    """Tool for comparing governance metrics between companies."""

    name: str = "governance_comparison_tool"
    description: str = """
    Compare and analyze Governance metrics between companies.
    Use this tool to analyze:
    1. Board composition
    2. Business ethics
    3. Transparency
    4. Risk management
    """
    args_schema: type[BaseModel] = ComparisonSchema
    llm: BaseChatModel

    def _run(
        self, target_data: str, comparison_data: str, context: Optional[str] = None
    ) -> dict:
        prompt = f"""Instructions:
        You are tasked with analyzing and comparing the governance performance of the target company against multiple competitors using the provided ESG data. The goal is to conduct a comprehensive analysis that automatically identifies common governance metrics between the target company and comparison companies, and provides both quantitative comparisons and qualitative insights on those metrics. If some metrics are unique to certain companies, use those as additional insights to enhance the report.

        <target_company>
        {target_data}
        </target_company>

        <comparison_companies>
        {comparison_data}
        </comparison_companies>

        {f'Additional context: {context}' if context else ''}
        Key Analysis Requirements:
        1. Identify Common Governance Metrics:
        Automatically identify the common governance metrics across the target company and the comparison companies (e.g., board diversity, business ethics, risk management, transparency).
        2. Compare these metrics quantitatively:
        Compare quantitative data for common governance metrics (e.g., percentage of board diversity, number of compliance violations, risk management scores, etc.).
        3. Analyze and Compare:
        Perform quantitative comparisons of common governance metrics between the target company and competitors.
        For metrics unique to the target company or comparison companies, provide qualitative insights that help explain how these unique factors contribute to the overall governance performance.
        4. Provide Key Insights:
        Focus on how the target company stands out in specific governance metrics compared to peers.
        Where unique data exists for any company, analyze how these unique governance initiatives or data points may impact the company’s overall governance performance or industry position.
        5. Gap Identification:
        Identify governance metrics where the target company lags behind.
        Identify governance metrics where the target company leads.
        Identify areas where the target company could improve or invest more to align with industry standards which will help in improving its governance performance.
        6. Identify Areas Where the Target Company Leads or Lags Behind:
        Identify areas where the target company excels or has the potential to improve relative to competitors in key governance metrics.

        Provide Your Analysis in This Format:
        {{
        "Target_Company_Analysis": {{
        "Strengths": [
        "Areas where the target company leads in specific governance metrics."
        ],
        "Weaknesses": [
        "Areas needing improvement or metrics where the target company lags behind competitors in terms of governance performance."
        ],
        "Competitive_Position": "Overall position of the target company compared to industry peers, including both strengths and gaps in governance performance.",
        "Unique_Initiatives": [
        "Distinctive governance practices or programs that set the target company apart in terms of board composition, business ethics, transparency, or risk management."
        ]
        }},
        "Comparative_Analysis": {{
        "Common_Metrics_Comparison": {{
        "Metric_1": {{
        "Target_Company": "Numerical value or description of governance performance",
        "Competitor_1": "Comparison value from competitor",
        "Competitor_2": "Comparison value from competitor",
        "Gap_Analysis": "Analysis of gaps in board composition and diversity between the target company and competitors."
        }},
        "Metric_2": {{
        "Target_Company": "Numerical value or description of governance performance",
        "Competitor_1": "Comparison value from competitor",
        "Competitor_2": "Comparison value from competitor",
        "Gap_Analysis": "Analysis of gaps in business ethics and compliance practices between the target company and competitors."
        }}
        }},
        "Industry_Leaders": [
        "Companies leading in specific governance metrics, based on available data."
        ],
        "Best_Practices": [
        "Best governance practices in the industry that the target company can emulate to improve its governance performance."
        ],
        "Gap_Analysis": "In-depth analysis of the gaps between the target company and the leading companies in governance metrics. This includes areas where the target company excels and areas where it needs to catch up."
        }},
        "Recommendations": {{
        "Short_Term": [
        "Immediate actions for improving performance in specific governance metrics where the target company lags."
        ],
        "Long_Term": [
        "Strategic governance initiatives and long-term goals that the target company should pursue to improve its governance performance."
        ],
        "Best_Practice_Adoption": [
        "Recommendations on specific best practices from leading companies that the target company should consider implementing to enhance its governance."
        ]
        }}
        }}
        Replace Target_Company with the name of the target company.
        Replace Competitor_1 and Competitor_2 with the names of the comparison companies according to the data you have collected.
        """

        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )
        return json.loads(repair_json(response.content))


class ESGAggregatorTool(BaseTool):
    """Tool for aggregating ESG analysis from individual components."""

    name: str = "esg_aggregator_tool"
    description: str = """
    Aggregate and synthesize ESG analysis from environmental, social, and governance components.
    Use this tool to:
    1. Combine individual ESG component analyses
    2. Generate overall ESG rankings
    3. Provide comprehensive recommendations
    """
    args_schema: type[BaseModel] = AggregatorSchema
    llm: BaseChatModel

    def _run(
        self,
        env_analysis: dict,
        social_analysis: dict,
        gov_analysis: dict,
        target_company_name: str,
    ) -> dict:
        prompt = f"""Instructions:
        You are tasked with synthesizing the ESG analyses for the target company from three key dimensions—Environmental, Social, and Governance—to produce a comprehensive report. This report should focus on the target company's ESG position, highlight strengths, identify gaps, and provide actionable recommendations. Compare the target company's performance against its peers, and include insights on how the company can improve and lead in various ESG metrics.

        <environmental_analysis>
        {json.dumps(env_analysis, indent=2)}
        </environmental_analysis>

        <social_analysis>
        {json.dumps(social_analysis, indent=2)}
        </social_analysis>

        <governance_analysis>
        {json.dumps(gov_analysis, indent=2)}
        </governance_analysis>

        Provide a comprehensive analysis focusing on {target_company_name}'s position in this format:
        {{
        "Executive_Summary": {{
        "Overall_Position": "Summary of {target_company_name}'s overall ESG performance across environmental, social, and governance aspects, with reference to the latest available data.",
        "Key_Strengths": [
        "Major competitive advantages, including top-performing ESG metrics and unique programs (e.g., renewable energy investments, high workforce diversity, strong ethics compliance)."
        ],
        "Critical_Gaps": [
        "Key areas for improvement based on quantitative comparisons and qualitative assessments (e.g., higher carbon emissions, lower community engagement, board diversity gaps, etc.)."
        ]
        }}
        }}
        Explanation of the Report Format:
        Executive_Summary:

        Overall_Position: Provides a high-level summary of target company’s ESG performance across environmental, social, and governance aspects.
        Key_Strengths: Identifies specific strengths of the target company, such as top ESG metrics or unique programs.
        Critical_Gaps: Highlights areas where the target company is lagging behind, based on both quantitative data (e.g., emissions) and qualitative assessments (e.g., board diversity).

        Key Considerations:
        Quantitative Data: Focus on numerical comparisons for common ESG metrics (e.g., emissions, turnover rates, board diversity percentages).
        Qualitative Insights: For unique metrics, provide contextual explanations about the company's ESG initiatives, even if direct numerical comparisons aren't available.
        Gap Analysis: The report explicitly highlights gaps between the target company and industry leaders, and how the target company can catch up or lead in certain ESG areas.
        """

        response = self.llm.invoke(
            [{"role": "system", "content": prompt, "temperature": 0.0}]
        )
        return json.loads(repair_json(response.content))


class ESGAnalyst:
    """Main class for ESG analysis."""

    def __init__(self, llm: BaseChatModel, temp_files: Optional[List[str]] = []):
        """Initialize the ESG analyst with required tools and files."""
        if not isinstance(llm, BaseChatModel):
            raise ValueError("llm must be an instance of BaseChatModel")
        self.llm = llm
        self.temp_files = [f for f in temp_files]

        # Initialize tools
        self.target_company_finder_tool = TargetCompanyFinderTool(llm=llm)
        self.search_tool = SearchTool(llm=llm)
        self.env_tool = EnvironmentComparisonTool(llm=llm)
        self.social_tool = SocialComparisonTool(llm=llm)
        self.gov_tool = GovernanceComparisonTool(llm=llm)
        self.aggregator_tool = ESGAggregatorTool(llm=llm)

    def get_esg_analysis_tool(self):
        """Creates a structured tool for ESG analysis."""

        def analyze_esg_content(query: str, context: Optional[str] = None) -> dict:
            try:
                # Load all company data
                all_companies_data = self._load_company_data()
                if not all_companies_data:
                    raise ValueError("No company data available for analysis")

                # Get target company name
                target_company = self.target_company_finder_tool.get_target_company(
                    query
                )
                logger.info(f"Target company: {target_company}")

                # Separate target and comparison data
                target_data, comparison_data = self._fetch_company_data(
                    target_company, all_companies_data
                )

                # Parallelize search queries
                search_queries = self.search_tool.get_company_search_queries(
                    target_company
                )
                with ThreadPoolExecutor() as executor:
                    search_future = executor.submit(
                        self.search_tool._run, search_queries
                    )
                    search_analysis = search_future.result()

                # Parallelize ESG analysis components
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit all analysis tasks
                    env_future = executor.submit(
                        self.env_tool._run, target_data, comparison_data, context
                    )
                    social_future = executor.submit(
                        self.social_tool._run, target_data, comparison_data, context
                    )
                    gov_future = executor.submit(
                        self.gov_tool._run, target_data, comparison_data, context
                    )

                    # Get results as they complete
                    futures = {
                        env_future: "Environmental",
                        social_future: "Social",
                        gov_future: "Governance",
                    }

                    analysis_results = {}
                    for future in as_completed(futures):
                        section = futures[future]
                        try:
                            analysis_results[section] = future.result()
                        except Exception as e:
                            logger.error(f"Error in {section} analysis: {str(e)}")
                            analysis_results[section] = {"error": str(e)}

                # Run aggregator tool with collected results
                aggregator_analysis = self.aggregator_tool._run(
                    analysis_results["Environmental"],
                    analysis_results["Social"],
                    analysis_results["Governance"],
                    target_company,
                )

                # Process search results into a clean summary
                search_summary = {
                    "Company_Overview": search_analysis.get("query_1", []),
                    "Industry_Classification": search_analysis.get("query_2", []),
                    "Recent_Initiatives": search_analysis.get("query_3", []),
                }

                # Create final combined results
                results = {
                    "Introduction": search_summary,
                    "Executive_Summary": aggregator_analysis["Executive_Summary"],
                    "ESG_Analysis": {
                        "Environmental": analysis_results["Environmental"],
                        "Social": analysis_results["Social"],
                        "Governance": analysis_results["Governance"],
                    },
                }

                # Generate visualizations
                charts = self.create_esg_visualizations(results)

                # Generate PDF report
                report_generator = ReportGenerator(charts)
                pdf_path = report_generator.generate_pdf(target_company, results)

                logger.info("ESG analysis completed successfully")
                logger.info(results)
                # return {"results": results, "pdf_path": pdf_path}
                return {
                    "message": json.dumps(results),
                    "metadata": {
                        "pdf_path": pdf_path,
                    },
                }

            except Exception as e:
                logger.error(f"Error in ESG analysis: {str(e)}")
                return {"error": str(e)}

        return StructuredTool.from_function(
            name="esg_analyst",
            func=analyze_esg_content,
            description="""Tool for analyzing Environmental, Social, and Governance (ESG) aspects of companies.""",
        )

    def _load_company_data(self) -> List[Dict]:
        """Load data from all available JSON files."""
        all_companies_data = []
        try:
            for file_path in self.temp_files:
                if file_path.endswith(".pdf"):
                    # Parse PDF and convert to text
                    pdf_parser = PDFParser()
                    text_content = pdf_parser.get_pdf_text(file_path)
                    # Convert text to JSON using TextToJSONConverter class
                    text_to_json = TextToJSONConverter(self.llm)
                    company_data = text_to_json.convert_to_json(text_content)
                    all_companies_data.append(company_data)
                elif file_path.endswith(".json"):
                    with open(file_path, "r") as file:
                        company_data = json.load(file)
                        all_companies_data.append(company_data)
                else:
                    logger.error(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
        return all_companies_data

    def _fetch_company_data(
        self, target_company_name: str, all_companies_data: List[list]
    ) -> Tuple[list, list]:
        """
        Separate target company data from comparison companies data.

        Args:
            target_company_name: Name of the target company
            all_companies_data: List of all company data dictionaries

        Returns:
            Tuple containing (target_company_data, comparison_companies_data)
        """
        target_data = None
        comparison_data = []

        for company_data in all_companies_data:
            # Use LLM to check if company names match
            check_prompt = f"Does '{company_data[0]['Company']}' refer to the same company as '{target_company_name}'? Check for spelling errors and other minor differences. Do not make any assumptions. Answer only 'yes' or 'no'."
            response = self.llm.invoke(
                [{"role": "system", "content": check_prompt}],
                temperature=0.0,
                model="gpt-4o-mini",
            )
            if response.content.strip().lower() == "yes":
                target_data = json.dumps(company_data)
            else:
                comparison_data.append(company_data)

        if not target_data:
            raise ValueError(f"Target company {target_company_name} not found in data")

        return target_data, json.dumps({"comparison_companies": comparison_data})

    def create_esg_visualizations(self, esg_data: dict) -> list[str]:
        """Create visualizations from ESG analysis data."""
        logger.info("Starting creation of ESG visualizations")
        try:
            logger.debug("Initializing GraphTool")
            graph_tool = GraphTool(self.llm)
            chart_files = []

            # Extract environmental metrics
            logger.debug("Extracting environmental metrics")
            env_analysis = esg_data.get("ESG_Analysis", {}).get("Environmental", {})
            metrics_comparison = env_analysis.get("Comparative_Analysis", {}).get(
                "Common_Metrics_Comparison", {}
            )

            # Create environmental visualizations
            logger.info("Creating environmental visualizations")
            for metric_name, metric_data in metrics_comparison.items():
                if isinstance(metric_data, dict):
                    # Skip if no numerical data
                    if not any(
                        isinstance(v, (int, float))
                        for v in metric_data.values()
                        if v not in ["No data", "Data not available"]
                    ):
                        logger.debug(
                            f"Skipping visualization for {metric_name} due to lack of numerical data"
                        )

                    query = (
                        f"Create a bar chart comparing {metric_name} between companies"
                    )
                    context = [f"{metric_name}:"]
                    for company, value in metric_data.items():
                        if company != "Gap_Analysis" and value not in [
                            "No data",
                            "Data not available",
                        ]:
                            context[0] += f" {company}: {value},"

                    if len(context[0]) > len(f"{metric_name}:"):
                        # Only create visualization if we have data
                        logger.debug(f"Creating visualization for {metric_name}")
                        file = graph_tool._run(query, context)
                        chart_files.append(file["metadata"]["image_path"])

            # Extract social metrics
            logger.debug("Extracting social metrics")
            social_analysis = esg_data.get("ESG_Analysis", {}).get("Social", {})
            metrics_comparison = social_analysis.get("Comparative_Analysis", {}).get(
                "Common_Metrics_Comparison", {}
            )

            # Create social visualizations
            logger.info("Creating social visualizations")
            for metric_name, metric_data in metrics_comparison.items():
                if isinstance(metric_data, dict):
                    # Skip if no numerical data
                    if not any(
                        isinstance(v, (int, float))
                        for v in metric_data.values()
                        if v not in ["No data", "Data not available"]
                    ):
                        logger.debug(
                            f"Skipping visualization for {metric_name} due to lack of numerical data"
                        )

                    query = (
                        f"Create a bar chart comparing {metric_name} between companies"
                    )
                    context = [f"{metric_name}:"]
                    for company, value in metric_data.items():
                        if company != "Gap_Analysis" and value not in [
                            "No data",
                            "Data not available",
                        ]:
                            context[0] += f" {company}: {value},"

                    if len(context[0]) > len(
                        f"{metric_name}:"
                    ):  # Only create visualization if we have data
                        logger.debug(f"Creating visualization for {metric_name}")
                        file = graph_tool._run(query, context)
                        chart_files.append(file["metadata"]["image_path"])

            # Extract governance metrics
            logger.debug("Extracting governance metrics")
            gov_analysis = esg_data.get("ESG_Analysis", {}).get("Governance", {})
            metrics_comparison = gov_analysis.get("Comparative_Analysis", {}).get(
                "Common_Metrics_Comparison", {}
            )

            # Create governance visualizations
            logger.info("Creating governance visualizations")
            for metric_name, metric_data in metrics_comparison.items():
                if isinstance(metric_data, dict):
                    # Skip if no numerical data
                    if not any(
                        isinstance(v, (int, float))
                        for v in metric_data.values()
                        if v not in ["No data", "Data not available"]
                    ):
                        logger.debug(
                            f"Skipping visualization for {metric_name} due to lack of numerical data"
                        )

                    query = (
                        f"Create a bar chart comparing {metric_name} between companies"
                    )
                    context = [f"{metric_name}:"]
                    for company, value in metric_data.items():
                        if company != "Gap_Analysis" and value not in [
                            "No data",
                            "Data not available",
                        ]:
                            context[0] += f" {company}: {value},"

                    if len(context[0]) > len(
                        f"{metric_name}:"
                    ):  # Only create visualization if we have data
                        logger.debug(f"Creating visualization for {metric_name}")
                        file = graph_tool._run(query, context)
                        chart_files.append(file["metadata"]["image_path"])

            logger.info("ESG visualizations created successfully")
            return chart_files

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}", exc_info=True)
            raise

    def create_esg_analysis_agent(self):
        """
        Creates an agent specifically for ESG analysis.

        Returns:
            AgentExecutor: Configured agent for ESG analysis.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a specialized ESG analysis assistant.
            Your task is to analyze companies' Environmental, Social, and Governance performance and provide comprehensive insights.""",
                ),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        tools = [
            self.target_company_finder_tool,
            self.search_tool,
            self.env_tool,
            self.social_tool,
            self.gov_tool,
            self.aggregator_tool,
        ]

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools)


if __name__ == "__main__":
    logger.info("\n=== Starting ESG Analysis ===")

    # Initialize with GPT-4 model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=config.OPENAI_API_KEY)
    logger.info("Initialized LLM and ESG Agent")

    # List of PDF files to analyze
    temp_files = [
        "pdf_files/asm_international.pdf",
        "pdf_files/atlas_copco.pdf",
        "pdf_files/yaskawa.pdf",
    ]

    esg_analyst = ESGAnalyst(llm, temp_files)
    esg_tool = esg_analyst.get_esg_analysis_tool()

    logger.info("\nAnalyzing companies...")

    # Run the analysis (now supports both sync and async)
    results = esg_tool.invoke(
        {
            "query": "Perform an ESG analysis on Atlas Copco against its peers ASM International and Yaskawa",
            "context": "Compare target company's ESG performance against industry peers",
        }
    )
    # Save results to JSON file
    with open("esg_analysis.json", "w") as file:
        json.dump(results, file, indent=2)

    logger.info("Analysis saved to esg_analysis.json")
    logger.info("Analysis completed successfully")
