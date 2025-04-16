# Import required libraries for file handling and encoding
import base64
import logging
import os
from pathlib import Path
from typing import Optional, Union

# Import libraries for date handling and report generation
import datetime
import markdown
from markdown.extensions.tables import TableExtension
from utils.equity_generation_utils.report_gen_chart_tool import ReportGenerator
from weasyprint import HTML

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_particulars_table(particulars: dict) -> str:
    """Creates compact HTML table for particulars with inline label-value pairs
    
    Args:
        particulars (dict): Dictionary containing key-value pairs of financial metrics
        
    Returns:
        str: HTML formatted table string
    """
    particulars_html = """
    <div class="particulars-section">
        <table class="particulars-table">
            <tbody>"""

    for key, value in particulars.items():
        particulars_html += f"""
            <tr>
                <td class="compact-cell">
                    <div class="particular-item">
                        <span class="particular-label">{key} : </span>
                        <strong class="particular-value">{value}</strong>
                    </div>
                </td>
            </tr>"""

    particulars_html += """
            </tbody>
        </table>
    </div>"""
    return particulars_html


def create_chart_layout(
    particulars_html: str, company_name: str, chart_intro_text: Optional[str] = None
) -> str:
    """Creates horizontal layout with large chart on left and very compact particulars on right
    
    Args:
        particulars_html (str): HTML string containing financial particulars table
        company_name (str): Name of the company
        chart_intro_text (Optional[str]): Custom introduction text for the chart
        
    Returns:
        str: HTML formatted layout string
    """
    try:
        report_gen = ReportGenerator()
        chart_bytes = report_gen.generate_stock_chart(company_name)
        if chart_bytes:
            chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")
            logger.info("Generated stock chart successfully")
            chart_html = f'<img src="data:image/png;base64,{chart_b64}" alt="Stock Chart" style="width: 100%; height: auto;">'
        else:
            logger.warning("No chart data generated")
            chart_html = ""
    except Exception as e:
        logger.error(f"Failed to generate chart: {str(e)}")
        chart_html = ""

    # Use provided intro text or fall back to default
    default_intro = f"""The chart below illustrates the historical stock performance of <strong>{company_name}</strong>, 
    accompanied by key financial particulars. This visualization provides insights into the company's market 
    trends and fundamental metrics that support our investment thesis."""

    # Convert intro text from markdown to HTML with TableExtension
    intro_text = chart_intro_text if chart_intro_text else default_intro
    intro_html = (
        markdown.markdown(intro_text, extensions=[TableExtension()])
        if chart_intro_text
        else intro_text
    )

    markdown_intro = f"""
    <div class="markdown-section">
        <h2>Stock Information</h2>
        <div class="report-content">
            <p>{intro_html}</p>
        </div>
    </div>"""

    return f"""
    {markdown_intro}
    <div style="display: flex; gap: 5px; align-items: start; margin-bottom: 20px;">
        <div class="chart-container" style="flex: 1;">
            {chart_html}
        </div>
        <div class="particulars-container" style="flex: 0 0 150px;">
            {particulars_html}
        </div>
    </div>"""


def generate_equity_report(
    company_name: str,
    recommendation: str,
    cmp: Union[str, float],
    target: Union[str, float],
    target_period: str,
    particulars: dict,
    body_markdown: str,
    chart_intro_text: Optional[str] = None,
    sector: str = "N/A",
    industry: str = "N/A",
    country: str = "N/A",
    exchange: str = "N/A",
):
    """Generates a complete equity research report in HTML format
    
    Args:
        company_name (str): Name of the company
        recommendation (str): Investment recommendation (BUY/HOLD/SELL)
        cmp (Union[str, float]): Current market price
        target (Union[str, float]): Target price
        target_period (str): Time period for target price
        particulars (dict): Financial metrics and details
        body_markdown (str): Main report content in markdown format
        chart_intro_text (Optional[str]): Custom chart introduction
        sector (str): Company sector
        industry (str): Company industry
        country (str): Company country
        exchange (str): Stock exchange
        
    Returns:
        str: Complete HTML template for the report
    """
    logger.info(f"Generating equity report for {company_name}")

    try:
        # Clean currency symbols and convert to float
        cmp = float(str(cmp).replace("₹", "").replace("$", "").replace(",", ""))
        target = float(str(target).replace("₹", "").replace("$", "").replace(",", ""))
        pct_change = round(((target - cmp) / cmp) * 100)
        logger.debug(f"Calculated target price change: {pct_change}%")

        report_gen = ReportGenerator()
        chart_bytes = report_gen.generate_stock_chart(company_name)
        if chart_bytes:
            chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")
            logger.info("Generated stock chart successfully")
        else:
            logger.warning("No chart data generated")
    except Exception as e:
        logger.error(f"Failed to generate chart: {str(e)}")
        chart_b64 = None

    # Convert markdown to HTML with table extension
    body_html = markdown.markdown(body_markdown, extensions=[TableExtension()])
    logger.debug("Converted body markdown to HTML")

    # Set recommendation color
    rec_color = {
        "BUY": "#00B300",  # Green
        "HOLD": "#FFA500",  # Yellow/Orange
        "SELL": "#FF0000",  # Red
    }.get(recommendation, "#000000")

    # Create chart HTML if chart data exists
    if chart_b64:
        chart_html = f"""
        <div class="chart-container">
            <img src="data:image/png;base64,{chart_b64}" alt="Stock Chart" style="width: 100%; height: auto;">
        </div>"""
    else:
        chart_html = ""

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Research Report - {company_name}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            @page {{
                size: A4;
                margin-top: 1cm;
                margin-bottom: 1cm;
                margin-left: 0.5cm;
                margin-right: 0.5cm;
                @bottom-right {{
                    content: "Equity Research - {company_name} | Page " counter(page);
                    font-family: 'Inter', Arial, sans-serif;
                    font-size: 12px;
                    color: #000;  /* Ensure visibility */
                    padding-right: 0.5cm;
                }}
            }}

            body {{
                font-family: 'Inter', Arial, sans-serif;
                line-height: 1.4;
                background: white;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                margin-top: 0.5cm;
                margin-bottom: 1cm;
                margin-left: 0.5cm;
                margin-right: 0.5cm;
                text-align: justify;
            }}

            h2 {{
                color: #1a237e;  /* Dark blue */
                font-style: italic;
                margin-bottom: 5px;
                padding-bottom: 8px;  /* Add padding for spacing above underline */
                border-bottom: 1px solid rgba(26, 35, 126, 0.3);  /* Thin, semi-transparent underline */
                display: inline-block;  /* Makes underline only as wide as text */
                margin-top: 12px;
            }}

            em {{
                color: #1a237e;  /* Blue for italics */
            }}

            hr {{
                border: 0;
                height: 2px;  /* Increased thickness */
                background: #333;  /* Darker color */
                margin: 20px 0;
                opacity: 0.8;  /* Added opacity for better visibility */
            }}

            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 15px 0;
                font-family: "Roboto", "Open Sans", sans-serif;
                background-color: #f5f5f5;
                color: #333333;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 8px;
                overflow: hidden;
            }}

            th {{ 
                background-color: #1a237e;
                color: #ffffff;
                font-weight: 600;
                padding: 8px 15px;
                text-align: left;
                font-size: 14px;
            }}

            td {{ 
                padding: 6px 15px;
                border-bottom: 1px solid #e0e0e0;
                font-size: 13px;
            }}

            .top-banner {{
                background: linear-gradient(135deg, #00008B 0%, #4B0082 100%);
                color: white;
                padding: 12px 16px;
                width: 100%;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .company-title {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin-bottom: 16px;
            }}

            .company-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 20px;
            }}

            .company-name {{
                font-family: 'IBM Plex Sans', sans-serif;
                font-size: 24px;
                font-weight: 600;
                letter-spacing: -0.2px;
                margin-bottom: 0;
                flex: 1;
            }}

            .company-info {{
                font-size: 14px;
                opacity: 0.9;
                display: flex;
                gap: 24px;
                align-items: center;
            }}

            .company-info span {{
                display: flex;
                align-items: center;
            }}

            .company-info strong {{
                font-weight: 600;
                color: #ffffff;
                margin-left: 6px;
            }}

            .company-info label {{
                color: rgba(255, 255, 255, 0.85);
            }}

            .recommendation {{
                font-family: 'IBM Plex Sans', sans-serif;
                font-weight: 600;
                letter-spacing: 0.5px;
                font-size: 22px;
                color: {rec_color};
                padding: 4px 12px;
                border: 2px solid {rec_color};
                border-radius: 6px;
                background-color: rgba(255, 255, 255, 0.1);
                min-width: 80px;
                text-align: center;
            }}

            .stock-details {{
                font-size: 14px;
                letter-spacing: -0.1px;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                padding: 12px 16px;
            }}

            .stock-details-row {{
                display: flex;
                justify-content: flex-start;
                gap: 20px;
                margin-bottom: 8px;
            }}

            .stock-details-row:last-child {{
                margin-bottom: 0;
                }}

            .stock-detail-item {{
                display: inline-flex;
                align-items: center;
                min-width: 180px;
            }}

            .stock-detail-item span:first-child {{
                opacity: 0.85;
                font-weight: 500;
                min-width: 80px;
            }}

            .stock-detail-item span:last-child {{
                font-weight: 600;
                margin-left: 8px;
            }}

            .content-wrapper {{
                padding: 20px;
                margin-bottom: 0;
            }}

            .main-content {{
                flex: 1;
                width: 100%;
                text-align: justify;
                margin-bottom: 0;
            }}

            .chart-container {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 8px;
                margin: 13px 0;
                margin-right: 10px;
                margin-left: -10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                width: 100%;
            }}
            
            .particulars-section {{
                margin: 0;
                font-family: 'IBM Plex Sans', sans-serif;
            }}

            .particulars-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                font-size: 11px;
                background: white;
            }}

            .compact-cell {{
                padding: 4px 6px !important;
                line-height: 1.4;
                height: 18px;
                border-bottom: 1px solid #f0f0f0;
                transition: background-color 0.2s;
            }}

            .compact-cell:hover {{
                background-color: #f8f9fa;
            }}

            .particular-item {{
                display: flex;
                align-items: center;
                gap: 6px;
                white-space: nowrap;
                height: 100%;
            }}

            .particular-label {{
                font-weight: 500;
                color: #444;
                font-size: 11px;
                letter-spacing: 0.2px;
            }}

            .particular-value {{
                color: #1a237e;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.1px;
            }}

            /* Add zebra striping for better readability */
            .particulars-table tr:nth-child(even) {{
                background-color: #fafbff;
            }}

            /* Style the first row differently */
            .particulars-table tr:first-child .compact-cell {{
                background-color: #f5f6fa;
                border-bottom: 2px solid #e0e0e0;
            }}

            .particulars-container {{
                flex: 0 0 150px;
                max-height: 300px;
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: #cbd5e0 #f8f9fa;
            }}

            /* Custom scrollbar styling */
            .particulars-container::-webkit-scrollbar {{
                width: 6px;
            }}

            .particulars-container::-webkit-scrollbar-track {{
                background: #f8f9fa;
            }}

            .particulars-container::-webkit-scrollbar-thumb {{
                background-color: #cbd5e0;
                border-radius: 3px;
            }}
            
            .markdown-body {{
                font-family: 'Inter', sans-serif;
                line-height: 1.6;
                color: #24292e;
                margin-top: 0;
                padding: 0 15px;
                font-size: 11pt;
            }}

            .content-pages {{
                margin-top: 0;
                padding-top: 0;
            }}

            /* Make all content use the same font size */
            .report-content, 
            .markdown-section p,
            .markdown-section li,
            .body-content p,
            .body-content li,
            .markdown-body p,
            .markdown-body li,
            .main-content p,
            .main-content li {{
                font-size: 11pt !important;
                line-height: 1.5;
                color: #333;
                font-family: Arial, sans-serif;
            }}

            /* Ensure headings are also consistent */
            .markdown-section h1,
            .markdown-section h2,
            .markdown-section h3,
            .markdown-body h1,
            .markdown-body h2,
            .markdown-body h3 {{
                font-family: 'Inter', sans-serif;
                color: #1a237e;
            }}

            /* Style tables consistently */
            .report-content table,
            .markdown-section table,
            .body-content table,
            .markdown-body table {{
                font-size: 11pt !important;
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
            }}

            /* Ensure table cells are consistent */
            td, th {{
                font-size: 11pt !important;
            }}
        </style>
    </head>
    <body>
        <div class="content-wrapper">
            <div class="top-banner">
                <div class="company-title">
                    <div class="company-header">
                        <div class="company-name">{company_name}</div>
                        <div class="recommendation">{recommendation}</div>
                    </div>
                    <div class="company-info">
                        Sector: <strong>{sector}</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Industry: <strong>{industry}</strong>
                    </div>
                </div>
                <div class="stock-details">
                    <div class="stock-details-row">
                        <span class="stock-detail-item">
                            <span>CMP:  <strong>{cmp}</strong></span>
                        </span>
                        <span class="stock-detail-item">
                            <span>Target:  <strong>{target} (+{pct_change}%)</strong> </span>
                        </span>
                        <span class="stock-detail-item">
                            <span>Target Period:  <strong>{target_period}</strong> </span>
                        </span>
                    </div>
                </div>
            </div>

            <div class="main-content">
                {create_chart_layout(create_particulars_table(particulars), company_name, chart_intro_text)}
                <div class="markdown-body">
                    {body_html}
                </div>
            </div>
        </div>
    </body>
    </html>"""

    logger.info("Successfully generated HTML template")
    return html_template


def generate_pdf(company_name: str, html_content: str) -> str:
    """Converts HTML report to PDF format
    
    Args:
        company_name (str): Name of the company
        html_content (str): HTML formatted report content
        
    Returns:
        str: Path to the generated PDF file
    """
    logger.info(f"Generating PDF for {company_name}")

    # Create output directory if it doesn't exist
    os.makedirs("files", exist_ok=True)
    logger.debug("Created output directory")

    # Generate PDF using WeasyPrint
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"files/{company_name.replace(' ', '_')}_report_{timestamp}.pdf"
    HTML(string=html_content).write_pdf(output_path)
    logger.info(f"Successfully generated PDF at {output_path}")
    return output_path


def main():
    """Main function to demonstrate report generation with sample data"""
    logger.info("Starting main execution")

    # Example usage
    company_name = "Apple Inc"
    # Create ReportGenerator instance first
    report_gen = ReportGenerator()
    chart_bytes = report_gen.generate_stock_chart(company_name)
    chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")
    sample_particulars = {
        "Market Capitalization (Rs crore)": "3,112",
        "FY24 Gross Debt (Rs crore)": "623",
        "Revenues": "1,461",
        "EBITDA": "132",
        "EBITDA margin (%)": "13.2",
        "Net Profit": "44",
        "EPS (Rs)": "6.8",
        "P/E (x)": "71.3",
        "EV/EBITDA (x)": "19.1",
        "RoCE (%)": "20.5",
        "RoE (%)": "17.8",
    }

    # sample_trend_data = """{
    #   x: ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'],
    #   y: [350, 380, 420, 450, 470, 484],
    #   type: 'scatter',
    #   mode: 'lines',
    #   line: {
    #     color: '#3dc4fa',
    #     width: 2
    #   },
    #   fill: 'tonexty',
    #   fillcolor: 'rgba(61, 196, 250, 0.2)'
    # }"""

    # sample_risks = [
    #     "Delays in tenders of smart meters",
    #     "Increase in competition",
    #     "Volatility in raw material prices",
    #     "Working capital intensive business",
    # ]

    sample_body = f"""# Company Overview

    ### About the stock:
    HPL Electric & Power (HPL), incorporated in 1992, is among India's leading electric equipment manufacturer with a formidable presence across two major segments, 1) Metering & Systems and 2) Consumer & Industrials.

    Metering & systems segment contributed ~58% to total revenues as of FY24, while balance ~42% by consumer & industrials. Company has 7 manufacturing facilities (5 in Haryana & 2 in Himachal) and 2 R&D centers. In meters segment, company has an annual capacity of 11 million units.

    ### Investment Rationale:
    * **Strong order backlog:** The company has secured orders worth ₹1,850 crore in the smart meter segment, with additional tenders in pipeline. The smart meter segment is expected to see accelerated growth due to government initiatives and increasing demand for smart grid solutions.

    * **Comprehensive product portfolio:** The consumer segment has shown a growth of 28% YoY, driven by expansion in LED lighting, wires & cables, and switchgear products. The company's diverse product range and strong distribution network of over 25,000 retailers positions it well for sustained growth.

    * **Improving operational efficiency:** Operating margins have improved by 250 basis points through strategic initiatives including automation, vendor consolidation, and better working capital management. The company's focus on R&D and innovation has led to development of new high-margin products.

    ### Financial Performance:

    The company has demonstrated strong financial performance with:
    - Revenue growth of 15% CAGR over the last 3 years
    - EBITDA margins expanding from 11% to 13.2% 
    - Debt reduction of ₹150 crore in FY24
    - Strong cash flow generation with operating cash flow of ₹225 crore

    ### Industry Outlook:

    The electric equipment industry in India is poised for robust growth driven by:
    1. Government's push for smart metering through RDSS scheme
    2. Rising electricity consumption and grid modernization
    3. Growth in real estate and infrastructure development
    4. Make in India initiative boosting domestic manufacturing

    ### Competitive Advantages:

    HPL Electric maintains competitive edge through:
    - Strong R&D capabilities with over 100 patents
    - Established brand presence of 3 decades
    - End-to-end manufacturing capabilities
    - Pan-India distribution network
    - Quality certifications and approvals

    ### Growth Drivers:

    Key growth catalysts include:
    1. Smart meter order pipeline of ₹3,000+ crore
    2. Expansion in premium consumer products
    3. Entry into new geographical markets
    4. Focus on high-margin segments
    5. Operating leverage benefits

    ### Risks and Challenges:

    While the outlook is positive, key risks include:
    1. Raw material price volatility
    2. Working capital intensity
    3. Competitive pressures
    4. Regulatory changes
    5. Economic slowdown impact

    ### Management Commentary:

    Management has outlined following strategic priorities:
    - Targeting 20%+ revenue growth through expansion in smart meter segment and consumer products
    - Margin expansion through product mix improvement and operational efficiencies
    - Debt reduction and working capital optimization to strengthen balance sheet
    - Capacity expansion in smart meters to meet growing order pipeline
    - Strengthening retail presence across tier 2/3 cities
    - Focus on R&D to develop innovative products
    - Exploring export opportunities in emerging markets
    - Investment in digital capabilities and automation
    - Building strong talent pipeline
    - Commitment to ESG initiatives and sustainability

    The management remains confident about achieving these objectives given the strong industry tailwinds, robust order book, and operational improvements. They emphasized their focus on profitable growth while maintaining market leadership position. The company plans to fund its expansion through internal accruals while maintaining a healthy balance sheet.

    Recent management interactions indicate:
    - Smart meter segment expected to grow at 40%+ CAGR
    - Consumer segment targeting 25% growth led by premium products
    - EBITDA margin guidance of 14-15% for FY25
    - Capex plan of ₹200 crore over next 2 years
    - Target to reduce working capital days to 120 by FY25
    """

    logger.info("Generating equity report with sample data")

    custom_intro = "This custom chart shows the performance metrics and key indicators for our analysis..."

    html_content = generate_equity_report(
        company_name=company_name,
        recommendation="BUY",
        cmp=484,
        target=660,
        target_period="6-12",
        particulars=sample_particulars,
        body_markdown=sample_body,
        chart_intro_text=custom_intro,
    )

    generate_pdf(company_name, html_content)
    logger.info("Completed main execution")
    # Save HTML file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{company_name.lower().replace(' ', '_')}_report.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Saved HTML report to {output_file}")


# Entry point of the script
if __name__ == "__main__":
    main()
