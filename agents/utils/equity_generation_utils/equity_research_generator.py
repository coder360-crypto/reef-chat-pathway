import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from config import AgentsConfig as Config
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field
from utils.equity_generation_utils.equity_research_content import (
    EquityResearchQuestionProcessor,
)
from utils.equity_generation_utils.financial_metrics import (
    get_financial_metrics,
    get_stock_ticker,
)
from utils.equity_generation_utils.report_gen_chart_tool import ReportGenerator
from utils.equity_generation_utils.template_equity import (
    generate_equity_report,
    generate_pdf,
)
from utils.equity_generation_utils.cost_tracker import CostTracker
from langchain_community.callbacks import OpenAICallbackHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class EquityResearchSchema(BaseModel):
    """Schema for Equity Research Tool input.
    
    Args:
        company_name (str): Company name to generate report for
        use_cache (Optional[bool]): Whether to use cached data
    """

    company_name: str = Field(
        ..., description="The name of the company to generate research report for"
    )
    use_cache: Optional[bool] = Field(
        True, description="Whether to use cached data or generate fresh report"
    )


class EquityResearchGenerator:
    """Generates comprehensive equity research reports.
    
    Handles generation of different report sections including executive summary,
    investment rationale, financial analysis etc. Uses OpenAI's GPT models for
    content generation and includes data validation and rectification.
    """

    def __init__(self):
        self.report_sections: Dict[str, str] = {}
        self.question_processor = EquityResearchQuestionProcessor()
        self.config = Config()
        # self._load_env()

        self.rectifier_prompt = """
        Review this section and check for:
        - If most of the data is missing then and rephrase such that the missing data is not mentioned
        - There should not be tables with all data unavailable/missing 
        - Do not lose out any data or tables from context make it comprehensive
        - no introduction or conclusion in the section to the point analysis
        - no subpoints at all 
        - no headings or, heading is already present in the report
        - no subsections at all write in paragraphs 
        - try to display data in tables if possible
        - less textual content more numerical data and to the point analysis (numerically dense content)
        - Make sure you utilise all the data provided in the context and do not miss any important data
        - Redundant or repeated content then rephrase such that it is not redundant or repeated
        - Logical flow and consistency if the flow is not logical then rearrange the paragraphs such that the flow is logical
        - Completeness of analysis if the analysis is not complete then reflect on it and improve it
             
        If issues are found resolve them and improve the section
        All key points should be present in the section and no important information should be missing

        Revise the section and address these issues while maintaining:
        - Clear, concise writing and to the point
        - Professional financial analysis
        - Proper markdown formatting
        - Consistent style throughout

        Format the response in markdown with:
        - Pipe tables for numerical/financial data
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
        """

    def _generate_section_content(
        self, section: str, context: str, section_type: str, ticker: str = None
    ) -> str:
        """Generates content for a specific report section.
        
        Args:
            section (str): Name of the section
            context (str): Context data for generation
            section_type (str): Type of section (Executive Summary, Stock Info etc)
            ticker (str, optional): Stock ticker symbol
            
        Returns:
            str: Generated and improved section content
        """
        # Set the appropriate prompt based on section_type
        if section_type == "Executive Summary":
            prompt = self._create_section_executivesummary_prompt(
                section, context, ticker
            )
        elif section_type == "Stock Information":
            prompt = self._create_section_stockinfo_prompt(section, context, ticker)
        elif section_type == "Investment Rationale":
            prompt = self._create_section_rationale_prompt(section, context)
        elif section_type == "Rating and Target Price":
            prompt = self._create_section_target_price_prompt(section, context, ticker)
        elif section_type == "Appendix":
            prompt = self._create_section_appendix_prompt(section, context)
        elif section_type == "Conclusion":
            prompt = self._create_section_conclusion_prompt(section, context)
        elif section_type == "Concall Highlights":
            prompt = self._create_section_concall_highlights_prompt(section, context)
        else:
            raise ValueError(f"Unknown section type: {section_type}")

        try:
            cost_tracker = CostTracker()
            callback = OpenAICallbackHandler()

            chat = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE,
                callbacks=[callback],
            )

            response = chat.invoke([HumanMessage(content=prompt)])
            content = response.content

            # Track costs after the call
            cost_tracker.add_usage(
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )

            logger.info(f"Generated content for {section}: {content}")
            improved_content = self._rectify_and_improve(content)

            return improved_content

        except Exception as e:
            logger.error(
                f"Error generating report text for section {section}: {str(e)}"
            )
            raise

    def _create_section_executivesummary_prompt(
        self, section: str, ticker: str, context: str
    ) -> str:
        return f"""Based on the following context about {section} from the report, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        Guidelines:
        - Do not mention the datasource name like 10-K
        - try to display data in tables if possible
        - Use bullet points only for listing key metrics, findings or strategic points - do not use bullet points for the entire section - No Subpoints at all 
        - if there is no data available in context then use the metrics data provided above
        - Write in a formal, professional tone
        - Focus on facts, figures and analysis
        - Avoid generic introductions/conclusions
        - Include all relevant data from the provided context and metrics
        - Do not generate any headings or subheadings its already present in the report i just need the section content
        - Important content and numbers should be wrapped in italics using markdown (*text*)
        - mention sources and citations of the data in the context
        
        Context:
        {context}

        Format the response in markdown with:
        - Add proper units to the table and normalize the data when numbers are very large (like billions, millions, thousands)
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
        - If there are discrepancies between context and metrics, clearly state which values are being used and why
        """

    def _create_section_stockinfo_prompt(
        self, section: str, ticker: str, context: str
    ) -> str:
        metrics = get_financial_metrics(ticker, 2024)

        return f"""Based on the following context about {section} from the report and financial metrics, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report i just need the section content**
        Guidelines:
        - Do not mention the datasource name like 10-K
        - try to display data in tables if possible
        - Use bullet points only for listing key metrics, findings or strategic points - do not use bullet points for the entire section - No Subpoints at all 
        - Present numerical data in well-formatted tables generate table only if required (do not generate tables is no data is available)
        - if there is no data available in context then use the metrics data provided above
        - Write in a formal, professional tone
        - Focus on facts, figures and analysis
        - Avoid generic introductions/conclusions
        - Include all relevant data from the provided context and metrics
        - Do not generate any headings or subheadings its already present in the report i just need the section content
        - Important content and numbers should be wrapped in italics using markdown (*text*)
        - mention sources and citations of the data in the context
        
        Context:
        {context}

        Format the response in markdown with:
        - Pipe tables for numerical/financial data
        - Add proper units to the table and normalize the data when numbers are very large (like billions, millions, thousands)
        - try to display data in tables if possible
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
        - If there are discrepancies between context and metrics, clearly state which values are being used and why
        """

    def _create_section_rationale_prompt(self, section: str, context: str) -> str:
        return f"""Based on the following context about Investment Rationale from the report, write a professional report section that outlines key drivers of growth, valuation insights, competitive advantages, market opportunities, and potential catalysts supporting the stock recommendation.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report i just need the section content**

        Guidelines:
        - make it as comprehensive as possible
        - Do not lose out any data or tables from context make it comprehensive
        - Do not mention the datasource name like 10-K
        - Use bullet points only for listing key metrics, findings or strategic points (no subpoints)
        - Present numerical data in well-formatted tables only if required
        - If data is not available in context, indicate as not available (avoid placeholders like $X or X%)
        - Write in a formal, professional tone focused on facts and analysis
        - Avoid generic introductions/conclusions
        - Include all relevant data from context
        - Important content and numbers should use markdown italics (*text*)
        - Include sources and citations for data
        
        Context:
        {context}

        Format the response in markdown with:
        - Pipe tables for numerical data
        - Add proper units to the table and normalize the data when numbers are very large (like billions, millions, thousands)
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
    """

    def _create_section_target_price_prompt(
        self, section: str, context: str, ticker: str
    ) -> str:

        metrics = get_financial_metrics(ticker, 2024)

        return f"""Based on the following context about {section} from the report, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report i just need the section content**
        
        Guidelines:
        - make it as comprehensive as possible
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
        - Add proper units to the table and normalize the data when numbers are very large (like billions, millions, thousands)
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
        """

    def _create_section_conclusion_prompt(self, section: str, context: str) -> str:
        return f"""Based on the following context about {section} from the report, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report i just need the section content**
        
        Guidelines:
        - It should be short and concise describing the key points and takeaways from the report
        - It should not be generic and should be specific to the company

        Context:
        {context}
        
        Given the above context write a professional report section in markdown format
        """

    def _create_section_concall_highlights_prompt(self, section: str, context: str) -> str:
        return f"""Based on the following context about {section} from the report, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report i just need the section content**

        Guidelines:
        - Make it as comprehensive as possible
        - Appendix should have all the data in tables and no other content
        - Do not mention the datasource name like 10-K
        - Avoid generic introductions/conclusions
        - Include all relevant data from context

        Context:
        {context}
        Use data only from context if numbers are not present in context then mention not available 

        Format the response in markdown with:
        - Pipe tables for numerical/financial data
        - Add proper units to the table and normalize the data when numbers are very large (like billions, millions, thousands)
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content
        """

    def _create_section_appendix_prompt(self, section: str, context: str) -> str:
        return f"""Based on the following context about {section} from the report, write a professional financial report section.Based on the following context about {section} from the report, write a professional financial report section.
        Present key financial metrics, performance indicators and market analysis in a clear, structured format.
        **Do not generate any headings or subheadings its already present in the report, I just need the section content**

        Guidelines:
        - Make it as comprehensive as possible
        - Appendix should have all the data in tables and no other content
        - Do not mention the datasource name like 10-K
        - Avoid generic introductions/conclusions
        - Include all relevant data from context

        Context:
        {context}
        Use data only from context if numbers are not present in context then mention not available 

        Format the response in markdown with:
        - Pipe tables for numerical/financial data
        - Add proper units to the table and normalize the data when numbers are very large (like billions, millions, thousands)
        - Bullet points for key findings
        - Professional paragraph structure for analysis
        - Italics (*) for important numbers and content

        Example:
        | Revenue ($ billions) | 2023 | 2022 | 2021 | 2020 |
        |---------------------|------|------|------|------|
        | 10.5                | 9.8   | 8.7   | 7.6   | 6.5   |

        Generate tables with above format only for all the different sections in the appendix
        """

    def _rectify_and_improve(self, content: str) -> str:
        try:
            cost_tracker = CostTracker()
            callback = OpenAICallbackHandler()
            chat = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE,
                callbacks=[callback],
            )

            response = chat.invoke(
                [
                    SystemMessage(content=self.rectifier_prompt),
                    HumanMessage(content=content),
                ],
            )

            # Track costs after the call
            cost_tracker.add_usage(
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )

            return response.content
        except Exception as e:
            logger.error(f"Error in rectification phase: {str(e)}")
            return content

    def stock_info_rectify(self, content: str, ticker: str) -> str:
        metrics = get_financial_metrics(ticker, 2024)

        metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        print(metrics_str)

        rectifier_prompt = """
        Review this financial report section and remove all tables while preserving the key information in paragraph form. When removing tables:
        Context:
        {content}
        financial metrics:
        {metrics_str}   

        Include no tables in the final output, else the report will look cluttered and response will be invalided
        
        Guidelines:
        - If there is any discrepancy between the context and the metrics then use the context data
        - Convert tabular data into flowing narrative text
        - Integrate numerical data naturally into sentences
        - Maintain all important metrics and figures
        - Ensure logical flow between paragraphs
        - Keep the writing clear, concise and professional
        - Use italics (*) for key numbers and metrics
        - Preserve comprehensive analysis and insights
        - Remove any remaining bullet points or section headers
        
        
        The final output should be continuous paragraphs with no tables, while retaining all critical financial information in a readable narrative format.
        """

        try:
            callback = OpenAICallbackHandler()
            chat = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                api_key=self.config.OPENAI_API_KEY,
                base_url=self.config.OPENAI_API_BASE,
                callbacks=[callback],
            )
            response = chat.invoke(
                [
                    SystemMessage(content=rectifier_prompt),
                    HumanMessage(
                        content=f"Context:\n{content}\n\nFinancial metrics:\n{metrics_str}"
                    ),
                ]
            )
            cost_tracker = CostTracker()
            # Track costs after the call
            cost_tracker.add_usage(
                input_tokens=callback.prompt_tokens,
                output_tokens=callback.completion_tokens,
                cost=callback.total_cost,
            )
            return response.content
        except Exception as e:
            logger.error(f"Error in rectification phase: {str(e)}")
            return content

    def generate_report(self, company_name: str) -> str:
        """Generates complete equity research report.
        
        Args:
            company_name (str): Name of company to analyze
            
        Returns:
            str: Path to generated PDF report
        """
        # Start report generation process
        logger.info(f"Starting equity research report generation for {company_name}")

        try:
            # Get ticker and metrics first since they're needed for other operations
            ticker = get_stock_ticker(company_name)
            if ticker == "None":
                logger.error(f"No ticker found for {company_name}")
                raise ValueError(f"No ticker found for {company_name}")

            metrics = get_financial_metrics(ticker, 2024)
            if not metrics:
                raise ValueError(
                    f"Could not retrieve financial metrics for {company_name}"
                )

            particulars = {
                "Market-Cap": metrics.get("Market Capitalization", 0),
                "Gross Debt": metrics.get("FY24 Gross Debt", 0),
                "Revenues": metrics.get("Revenues", 0),
                "EBITDA": metrics.get("EBITDA", 0),
                "EBITDA margin (%)": metrics.get("EBITDA margin (%)", 0),
                "Net Profit": metrics.get("Net Profit", 0),
                "EPS": metrics.get("EPS (Rs)", 0),
                "P/E": metrics.get("P/E (x)", 0),
                "EV/EBITDA": metrics.get("EV/EBITDA (x)", 0),
                "RoCE(%)": metrics.get("RoCE (%)", 0),
                "RoE(%)": metrics.get("RoE (%)", 0),
            }

            # Parallelize content and final analysis retrieval
            with ThreadPoolExecutor(max_workers=2) as executor:
                content_future = executor.submit(
                    self.question_processor.get_content_analysis, company_name
                )
                final_future = executor.submit(
                    self.question_processor.get_final_analysis, company_name
                )

                content_analysis = content_future.result()
                final_analysis = final_future.result()

            # Parallelize section content generation
            sections_to_generate = [
                (
                    "Executive Summary",
                    content_analysis.get("EXECUTIVE_SUMMARY", []),
                    "Executive Summary",
                    ticker,
                ),
                (
                    "Investment Rationale",
                    content_analysis.get("INVESTMENT_RATIONALE", []),
                    "Investment Rationale",
                    None,
                ),
                (
                    "Market Overview",
                    content_analysis.get("MARKET_OVERVIEW", []),
                    "Investment Rationale",
                    None,
                ),
                (
                    "Recent News and Trends",
                    content_analysis.get("RECENT_NEWS_AND_TRENDS", []),
                    "Investment Rationale",
                    None,
                ),
                (
                    "Investment Thesis",
                    content_analysis.get("INVESTMENT_THESIS", []),
                    "Investment Rationale",
                    None,
                ),
                (
                    "Investment Analysis",
                    content_analysis.get("INVESTMENT_ANALYSIS", []),
                    "Investment Rationale",
                    None,
                ),
                (
                    "Financial Analysis",
                    content_analysis.get("FINANCIAL_ANALYSIS", []),
                    "Investment Rationale",
                    None,
                ),
                (
                    "Rating and Target Price",
                    final_analysis.get("RATING_AND_TARGET_PRICE", []),
                    "Rating and Target Price",
                    None,
                ),
                (
                    "Risk Analysis",
                    final_analysis.get("RISK_ANALYSIS", []),
                    "Rating and Target Price",
                    None,
                ),
                (
                    "Recommendations",
                    final_analysis.get("RECOMMENDATIONS", []),
                    "Rating and Target Price",
                    None,
                ),
                ("Appendix", final_analysis.get("APPENDIX", []), "Appendix", None),
                (
                    "Conclusion",
                    final_analysis.get("CONCLUSION", []),
                    "Conclusion",
                    None,
                ),
                (
                    "Concall Highlights",
                    content_analysis.get("CONCALL_HIGHLIGHTS", []),
                    "Concall Highlights",
                    None,
                ),
            ]

            section_results = {}
            with ThreadPoolExecutor(max_workers=12) as executor:
                future_to_section = {
                    executor.submit(
                        self._generate_section_content,
                        section_name,
                        "\n".join(content),
                        section_type,
                        ticker_arg,
                    ): section_name
                    for section_name, content, section_type, ticker_arg in sections_to_generate
                }

                for future in as_completed(future_to_section):
                    section_name = future_to_section[future]
                    try:
                        section_results[section_name] = future.result()
                    except Exception as e:
                        logger.error(f"Error generating {section_name}: {str(e)}")
                        section_results[section_name] = (
                            f"Error generating content: {str(e)}"
                        )

            # rating_target_price = final_analysis.get("RATING_AND_TARGET_PRICE", [])

            # Parallelize chart generation with other operations
            with ThreadPoolExecutor() as executor:
                chart_future = executor.submit(self._generate_chart, company_name)
                rec_future = executor.submit(
                    self.question_processor.extract_recommendation,
                    final_analysis,
                )

                try:
                    chart_b64 = chart_future.result()
                except Exception as e:
                    logger.error(f"Failed to generate chart: {str(e)}")
                    chart_b64 = None

                try:
                    rec = rec_future.result()
                except Exception as e:
                    logger.error(
                        f"Failed to extract recommendation and risks: {str(e)}"
                    )
                    rec = ""

            # Assemble the report content
            body_markdown = self._assemble_report_content(section_results)

            # Generate HTML and PDF
            html_content = generate_equity_report(
                company_name=f"{company_name} ({ticker})",
                recommendation=rec,
                cmp=metrics.get("Current Price", 484),
                target=metrics.get("Target Price", 0),
                target_period=metrics.get("Target Period", "N/A"),
                particulars=particulars,
                body_markdown=body_markdown,
                sector=metrics.get("Sector", "N/A"),
                industry=metrics.get("Industry", "N/A"),
                chart_intro_text=section_results.get("Executive Summary", ""),
            )

            pdf_path = generate_pdf(company_name, html_content)
            logger.info(f"Generated equity research report at {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"Unexpected error generating report: {str(e)}")
            raise

    def _generate_chart(self, company_name: str) -> Optional[str]:
        """Helper method to generate chart"""
        try:
            report_gen = ReportGenerator()
            chart_bytes = report_gen.generate_stock_chart(company_name)
            if chart_bytes:
                return base64.b64encode(chart_bytes).decode("utf-8")
            return None
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}")
            return None

    def _assemble_report_content(self, section_results: Dict[str, str]) -> str:
        """Helper method to assemble the report content from section results"""
        body_markdown = ""

        # Add Executive Summary
        if "Executive Summary" in section_results:
            body_markdown += section_results["Executive Summary"] + "\n\n"

        # Add Investment Rationale section
        body_markdown += "## Investment Rationale\n\n"
        for section in [
            "Investment Rationale",
            "Market Overview",
            "Recent News and Trends",
            "Investment Thesis",
            "Investment Analysis",
            "Financial Analysis",
        ]:
            if section in section_results:
                body_markdown += section_results[section] + "\n\n"

        # Add Rating and Target Price section
        body_markdown += "## Rating and Target Price\n\n"
        for section in ["Rating and Target Price", "Risk Analysis", "Recommendations"]:
            if section in section_results:
                body_markdown += section_results[section] + "\n\n"

        body_markdown += "## Concall Highlights\n\n"
        if "Concall Highlights" in section_results:
            body_markdown += section_results["Concall Highlights"] + "\n\n"

        # Add Conclusion
        body_markdown += "## Conclusion\n\n"
        if "Conclusion" in section_results:
            body_markdown += section_results["Conclusion"] + "\n\n"

        # Add Appendix
        if "Appendix" in section_results:
            body_markdown += "## Appendix\n\n"
            body_markdown += section_results["Appendix"] + "\n\n"

        return body_markdown


class EquityResearchTool(BaseTool):
    """Tool for generating equity research reports.
    
    Args:
        llm (Any): Language model instance
        generator (EquityResearchGenerator): Report generator instance
        
    Provides interface for generating comprehensive financial analysis and 
    research reports for companies using the EquityResearchGenerator.
    """

    name: str = "equity_research_tool"
    description: str = """
    Generate comprehensive equity research reports for companies. This tool:
    1. Retrieves company financial metrics and stock information
    2. Analyzes investment rationale and market position
    3. Provides rating and target price analysis
    4. Generates a complete PDF report with charts and appendices
    
    Use this tool when you need detailed financial analysis and research reports for companies.
    """
    args_schema: type[BaseModel] = EquityResearchSchema
    llm: Any = Field(default=None)
    generator: EquityResearchGenerator = Field(default_factory=EquityResearchGenerator)

    def __init__(self) -> None:
        """Initialize the equity research tool."""
        super().__init__()
        self.generator = EquityResearchGenerator()

    def _run(self, company_name: str, use_cache: bool = False, **kwargs) -> str:
        """Run the equity research tool."""
        try:
            # Reset cost tracker at the start of each run
            cost_tracker = CostTracker()
            cost_tracker.reset()

            print(f"[EquityResearchTool] Generating report for: {company_name}")

            pdf_path = self.generator.generate_report(company_name)

            # Get final cost metrics
            cost_metrics = cost_tracker.get_metrics()

            return {
                "message": f"Successfully generated equity research report at: {pdf_path}",
                "metadata": {"pdf_path": pdf_path, "cost_metrics": cost_metrics},
            }
        except Exception as e:
            logger.error(f"Error in equity research tool: {str(e)}")
            return {
                "message": f"Error generating research report: {str(e)}",
                "metadata": {},
            }

    async def _arun(self, company_name: str, use_cache: bool = True, **kwargs) -> str:
        """Async implementation of the tool."""
        return self._run(company_name, use_cache, **kwargs)


if __name__ == "__main__":
    # Track execution time
    start_time = time.time()

    research_tool = EquityResearchTool()
    result = research_tool._run(company_name="Cipla", use_cache=False)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
