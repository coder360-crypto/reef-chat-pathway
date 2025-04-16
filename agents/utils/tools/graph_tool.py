import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Union

import altair as alt
import chromedriver_autoinstaller
import pandas as pd
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, ConfigDict, Field
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)


class GraphSchema(BaseModel):
    """Schema for Graph Tool input.
    
    Args:
        query (str): The query describing what to visualize
        context (Optional[Union[List[str], str]]): Data context for creating the visualization
    """

    query: str = Field(..., description="The query describing what to visualize")
    context: Optional[Union[List[str], str]] = Field(
        None, description="Data context for creating the visualization"
    )


class ChartCode(BaseModel):
    """Schema for chart generation output.
    
    Args:
        code (str): Python/JavaScript code to generate the visualization
        reasoning (str): Explanation for the visualization choices
        chart_type (str): Type of chart (altair or highcharts)
        data (Optional[Union[dict, str]]): Extracted data in JSON format for Highcharts
    """

    code: str = Field(
        ..., description="Python/JavaScript code to generate the visualization"
    )
    reasoning: str = Field(..., description="Explanation for the visualization choices")
    chart_type: str = Field(
        "altair", description="Type of chart (altair or highcharts)"
    )
    data: Optional[Union[dict, str]] = Field(
        None, description="Extracted data in JSON format for Highcharts"
    )


class GraphTool(BaseTool):
    """Tool for generating data visualizations.
    
    Args:
        llm (Any): Language model for generating visualizations
        prompt (ChatPromptTemplate): Default prompt template
        altair_prompt (ChatPromptTemplate): Prompt template for Altair charts
        highcharts_prompt (ChatPromptTemplate): Prompt template for Highcharts
    """

    name: ClassVar[str] = "graph_tool"
    description: ClassVar[
        str
    ] = """
    Generate data visualizations based on queries and data context.
    Use this tool when you need to:
    1. Create charts and graphs from data
    2. Visualize trends and patterns
    3. Compare different data points
    4. Show distributions or relationships in data
    """
    args_schema: ClassVar[type] = GraphSchema
    llm: Any = Field(default=None)
    prompt: ChatPromptTemplate = Field(default=None)
    altair_prompt: ChatPromptTemplate = Field(default=None)
    highcharts_prompt: ChatPromptTemplate = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, llm: Any) -> None:
        """Initialize the graph tool."""
        # Create prompts first
        altair_system_prompt = """You are a data visualization expert. Given a query and context containing data, extract the relevant information and create a visualization using Vega-Altair.
            You must use the Altair library (import altair as alt) to create the visualization.

            Important: When processing data:
            1. Carefully preserve the scale and units of each metric when extracting values
            2. For multiple data series:
               - Maintain separate scales for metrics with different units
               - Use separate plots or dual axes for metrics with different scales
            3. Remove currency symbols and convert abbreviations (e.g., 'M' for million, 'K' for thousand) while preserving the original scale

            Important guidelines:
            1. Pay special attention to units in the data (e.g., percentages, currencies, temperatures, etc.)
            2. Always maintain the original units in axis labels and tooltips
            3. Create subplots or separate plots when:
            - Comparing multiple metrics with different scales
            - Showing related but distinct aspects of the data
            - Visualizing before/after scenarios
            4. Ignore any data series that contains missing values to ensure accurate visualization
            5. Depending on the data available you need to decide what kind of plot to make.
            6. Use the following color palette for all visualizations and try to maintain good contrast between colors, use other colors only if needed: ['#1B206E', '#232895', '#9CA3AF', '#5B77C8', '#2B3978', '#2D3870', '#1A2453']
            7. When appending data to a DataFrame, use pd.concat instead of the deprecated append method.

            Your response must be a valid JSON object with EXACTLY the following structure:
            {{
                "code": "string containing complete Python code to generate the plot",
                "reasoning": "string explaining your visualization choices"
            }}

            The code should:
            1. Create any necessary data structures (lists, DataFrames, etc.)
            2. Generate and display the plot using Altair
            3. Include appropriate titles and labels
            4. Handle the data cleaning/formatting
            5. Make sure the plot is well-designed and looks advanced
            6. Use subplots when dealing with multiple scales or metrics
            7. IMPORTANT: Ensure that x-axis labels are displayed horizontally for better readability.(use labelAngle=0, if applicable)
            8. IMPORTANT: If the labels are too long use appropriate abbreviations while displaying to avoid clutter.
            9. IMPORTANT: Always assign your final visualization to a variable named 'chart'

            NOTE : You don't need to save the chart as a PNG file. Just return the code.

            Example 1:
            Query: "Show me sales growth over time"
            Context: "Sales by year: 2020: $100K, 2021: $150K, 2022: $200K"
            Response:
            {{
                "code": "import altair as alt\\nimport pandas as pd\\n\\ndata = {'Year': ['2020', '2021', '2022'],'Sales': [100_000, 150_000, 200_000]}\\ndf = pd.DataFrame(data)\\n\\nchart = alt.Chart(df).mark_line().encode(\\n    x=alt.X('Year', axis=alt.Axis(labelAngle=0)),\\n    y=alt.Y('Sales:Q', axis=alt.Axis(format='$,.0f'))\\n).properties(\\n    title='Sales Growth Over Time',\\n    width=600,\\n    height=400\\n)\\n",
                "reasoning": "Used a line chart to show the temporal progression of sales values. Added proper currency formatting and clear labels."
            }}

            Example 2:
            Query: "Display temperature changes over a week"
            Context: "Temperature readings: Monday: 20°C, Tuesday: 22°C, Wednesday: 19°C, Thursday: 21°C, Friday: 23°C"
            Response:
            {{
                "code": "import altair as alt\\nimport pandas as pd\\n\\ndata = {'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'Temperature': [20, 22, 19, 21, 23]}\\ndf = pd.DataFrame(data)\\n\\nchart = alt.Chart(df).mark_line().encode(\\n    x=alt.X('Day:O', axis=alt.Axis(labelAngle=0)),\\n    y=alt.Y('Temperature:Q', axis=alt.Axis(title='Temperature (°C)'))\\n).properties(\\n    title='Temperature Changes Over a Week',\\n    width=600,\\n    height=400\\n)\\n",
                "reasoning": "Used a line chart to visualize temperature changes over the week. Ensured that day labels are displayed horizontally for better readability."
            }}

            Important: Make sure to assign your final chart to a variable named 'chart' before returning.
            For example:
            chart = alt.Chart(df).mark_line()...
    """

        highcharts_system_prompt = """You are a data visualization expert. Given a query and context containing data, extract the relevant information and create a visualization using Highcharts.

    Important: You must first extract numerical values from the context, removing any currency symbols and converting abbreviations.
    Important guidelines:
    1. Pay attention to units in the data (percentages, currencies, temperatures, etc.)
    2. Maintain original units in axis labels and tooltips
    3. Use advanced Highcharts features like:
    - 3D charts when appropriate
    - Interactive tooltips
    - Animations
    - Multiple series types
    - Custom themes
    4. Ignore any data series that contains missing values to ensure accurate visualization
    5. Depending on the data avilable you need to decide what kind of plot to make.
    6. Use the following color palette for all visualizations and try to maintain good contrast between colors , use other colors only if needed: ['#1B206E', '#232895', '#9CA3AF', '#5B77C8', '#2B3978', '#2D3870', '#1A2453']

    Your response must be a valid JSON object with EXACTLY the following structure:
    {{
        "code": "string containing complete JavaScript code for Highcharts",
        "reasoning": "string explaining your visualization choices",
        "data": "extracted data in JSON format"
    }}

    The code should:
    1. Include complete Highcharts configuration
    2. Use appropriate chart types and features
    3. Include proper titles, labels, and legends
    4. Implement interactive features
    5. Use professional color schemes
    6. Include responsive design options
    7. Implement proper data formatting
    8. Add helpful tooltips
    9. IMPORTANT: If the labels are too long use appropriate abbreviations while displaying to avoid clutter.

Example 1:
Query: "Show me sales growth over time"
Context: "Sales by year: 2020: $100K, 2021: $150K, 2022: $200K"
Response:
{{
    "code": "Highcharts.chart('container', {
        chart: {
            type: 'line',
            animation: true,
            style: {
                fontFamily: 'Arial, sans-serif'
            }
        },
        title: {
            text: 'Sales Growth Over Time',
            style: { fontSize: '20px' }
        },
        xAxis: {
            categories: ['2020', '2021', '2022'],
            title: { text: 'Year' }
        },
        yAxis: {
            title: { text: 'Sales ($)' },
            labels: {
                formatter: function() {
                    return '$' + this.value.toLocaleString();
                }
            }
        },
        series: [{
            name: 'Sales',
            color: '#1B206E',
            data: [100000, 150000, 200000],
            marker: {
                enabled: true,
                symbol: 'circle',
                radius: 6
            }
        }],
        tooltip: {
            formatter: function() {
                return '<b>' + this.x + '</b><br/>Sales: $' + 
                       this.y.toLocaleString();
            }
        },
        plotOptions: {
            line: {
                animation: {
                    duration: 2000
                }
            }
        }
    });",
    "reasoning": "Used a line chart with gradient colors and animations to show sales progression over time. Added currency formatting and interactive tooltips for better data interpretation.",
    "data": {{
        "years": ["2020", "2021", "2022"],
        "sales": [100000, 150000, 200000]
    }}
}}  

Example 2:
Query: "Create a 3D pie chart showing market distribution"
Context: "Market shares: Company A: 30%, Company B: 25%, Company C: 20%, Others: 25%"
Response:
{{
    "code": "Highcharts.chart('container', {
        chart: {
            type: 'pie',
            options3d: {
                enabled: true,
                alpha: 45,
                beta: 0
            }
        },
        title: {
            text: 'Market Share Distribution',
            style: { fontSize: '20px' }
        },
        plotOptions: {
            pie: {
                depth: 35,
                allowPointSelect: true,
                cursor: 'pointer',
                dataLabels: {
                    enabled: true,
                    format: '{point.name}: {point.percentage:.1f}%',
                    style: {
                        fontSize: '12px'
                    }
                },
                showInLegend: true
            }
        },
        series: [{
            name: 'Market Share',
            data: [
                {
                    name: 'Company A',
                    y: 30,
                    color: '#1B206E',
                    sliced: true,
                    selected: true
                },
                {
                    name: 'Company B',
                    y: 25,
                    color: '#232895'
                },
                {
                    name: 'Company C',
                    y: 20,
                    color: '#5B77C8'
                },
                {
                    name: 'Others',
                    y: 25,
                    color: '#9CA3AF'
                }
            ]
        }],
        tooltip: {
            pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
        },
        colors: ['#1B206E', '#232895', '#5B77C8', '#9CA3AF']
    });",
    "reasoning": "Used a 3D pie chart with interactive features to show market share distribution. Added percentage labels and tooltips for clarity.",
    "data": {{
        "companies": ["Company A", "Company B", "Company C", "Others"],
        "shares": [30, 25, 20, 25]
    }}  
}}

Example 3:
Query: "Show regulatory capital ratios comparison"
Context: "CET1 capital ratio: 10.3%, Tier 1 capital ratio: 11.1%, Total capital ratio: 12.8%"
Response:
{{
    "code": "Highcharts.chart('container', {
        chart: {
            type: 'column',
            options3d: {
                enabled: true,
                alpha: 15,
                beta: 15,
                depth: 50,
                viewDistance: 25
            }
        },
        title: {
            text: 'Regulatory Capital Ratios',
            style: { fontSize: '20px' }
        },
        xAxis: {
            categories: ['CET1 Ratio', 'Tier 1 Ratio', 'Total Capital Ratio'],
            labels: {
                style: { fontSize: '13px' }
            }
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Percentage (%)'
            },
            labels: {
                format: '{value}%'
            }
        },
        plotOptions: {
            column: {
                depth: 25,
                colorByPoint: true,
                dataLabels: {
                    enabled: true,
                    format: '{point.y:.1f}%',
                    style: { fontSize: '13px' }
                }
            }
        },
        series: [{
            name: 'Capital Ratios',
            data: [{
                y: 10.3,
                color: '#1B206E'
            }, {
                y: 11.1,
                color: '#232895'
            }, {
                y: 12.8,
                color: '#5B77C8'
            }],
            colorByPoint: true
        }],
        tooltip: {
            formatter: function() {
                return '<b>' + this.x + '</b><br/>' +
                       'Ratio: ' + this.y.toFixed(1) + '%';
            }
        },
        colors: ['#1B206E', '#232895', '#5B77C8']
    });",
    "reasoning": "Used a 3D column chart with gradient colors to compare different capital ratios. Added percentage labels and interactive tooltips for better data interpretation.",
    "data": {{
        "categories": ["CET1 Ratio", "Tier 1 Ratio", "Total Capital Ratio"],
        "values": [10.3, 11.1, 12.8]
    }}
}}

Example 4:
Query: "Compare regulatory capital ratios between American Express Company and American Express National Bank"
Context: "American Express Company - CET1: 10.3%, Tier 1: 11.1%, Total Capital: 12.8%, Leverage: 9.9%
American Express National Bank - CET1: 11.3%, Tier 1: 11.3%, Total Capital: 13.2%, Leverage: 9.7%"
Response:
{{
    "code": "Highcharts.chart('container', {
        chart: {
            type: 'column',
            options3d: {
                enabled: true,
                alpha: 15,
                beta: 15,
                depth: 50,
                viewDistance: 25
            }
        },
        title: {
            text: 'Regulatory Capital Ratios Comparison',
            style: { fontSize: '20px' }
        },
        xAxis: {
            categories: ['CET1 Ratio', 'Tier 1 Ratio', 'Total Capital Ratio', 'Leverage Ratio'],
            labels: {
                style: { fontSize: '13px' },
                rotation: 0
            }
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Percentage (%)',
                margin: 20
            },
            labels: {
                format: '{value}%'
            }
        },
        legend: {
            align: 'center',
            verticalAlign: 'bottom',
            layout: 'horizontal'
        },
        plotOptions: {
            column: {
                depth: 40,
                grouping: true,
                pointPadding: 0.05,
                groupPadding: 0.2,
                dataLabels: {
                    enabled: true,
                    format: '{point.y:.1f}%',
                    style: { fontSize: '11px' }
                }
            }
        },
        series: [{
            name: 'American Express Company',
            color: '#2F80ED', // Ensure legend uses the same color as the data points
            data: [{
                y: 10.3,
                color: '#2F80ED'
            }, {
                y: 11.1,
                color: '#2F80ED'
            }, {
                y: 12.8,
                color: '#2F80ED'
            }, {
                y: 9.9,
                color: '#2F80ED'
            }]
        }, {
            name: 'American Express National Bank',
            color: '#85B6F2', // Ensure legend uses the same color as the data points
            data: [{
                y: 11.3,
                color: '#85B6F2'
            }, {
                y: 11.3,
                color: '#85B6F2'
            }, {
                y: 13.2,
                color: '#85B6F2'
            }, {
                y: 9.7,
                color: '#85B6F2'
            }]
        }],
        tooltip: {
            formatter: function() {
                return '<b>' + this.x + '</b><br/>' +
                    this.series.name + ': ' + this.y.toFixed(1) + '%';
            }
        }
    });",
    "reasoning": "Used a 3D column chart with consistent color coding where each entity maintains its distinct shade across all metrics. American Express Company uses a darker blue (#2F80ED) and American Express National Bank uses a lighter blue (#85B6F2) consistently across all four metrics, making it easy to compare values between entities while maintaining visual grouping.",
    "data": {{
        "metrics": ["CET1 Ratio", "Tier 1 Ratio", "Total Capital Ratio", "Leverage Ratio"],
        "amex_company": [10.3, 11.1, 12.8, 9.9],
        "amex_bank": [11.3, 11.3, 13.2, 9.7]
    }}
}}


Additional color and labeling guidelines:
1. For pie/donut charts:
   - Use distinct colors for each segment
   - Always include data labels showing both category and percentage
   - Include a legend for better readability

2. For bar/column charts with multiple categories per label:
- When showing multiple categories under the same x-axis label:
    - bars should be placed adjacent to each other and not front and back
    - Use different shades of the same base color for all categories
    - Maintain consistent shading across all x-axis labels
    - Use these same shades for all labels on x-axis
- Include value labels on the bars/columns
- Position legend below the chart showing category distinctions
- Use proper grouping and spacing between metric groups
- Ensure tooltip shows both metric and category information

3. For line charts:
   - Use different colors and line styles for multiple series
   - Include markers for data points

4. General guidelines:
   - Ensure color contrast meets accessibility standards
   - Use consistent color coding across related metrics
   - Include clear labels and tooltips
   - Match colors to their semantic meaning (e.g., green for positive, red for negative)

"""

        # Create the prompt templates
        altair_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=altair_system_prompt),
                ("human", "{query}"),
                MessagesPlaceholder(variable_name="context", optional=True),
            ]
        )

        highcharts_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=highcharts_system_prompt),
                ("human", "{query}"),
                MessagesPlaceholder(variable_name="context", optional=True),
            ]
        )

        # Initialize the parent class with all required fields
        super().__init__(
            llm=llm,
            prompt=altair_prompt,  # Use altair prompt as default
            altair_prompt=altair_prompt,
            highcharts_prompt=highcharts_prompt,
        )

        # Set up Altair configuration
        self._setup_altair()

    def _setup_altair(self):
        """Configure Altair settings."""
        alt.data_transformers.disable_max_rows()
        alt.renderers.enable("default")

    def _generate_highcharts_visualization(self, result: ChartCode) -> tuple[str, str]:
        """Generate Highcharts visualization and save as PNG."""
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Highcharts Visualization</title>
            <script src="https://code.highcharts.com/highcharts.js"></script>
            <script src="https://code.highcharts.com/highcharts-3d.js"></script>
            <script src="https://code.highcharts.com/modules/exporting.js"></script>
            <script src="https://code.highcharts.com/modules/export-data.js"></script>
            <script src="https://code.highcharts.com/modules/accessibility.js"></script>
            <style>
                #container {{
                    min-width: 800px;
                    height: 600px;
                    margin: 0.5rem auto;
                }}
                .highcharts-credits {{
                    display: none !important;
                }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <script>
                // Global Highcharts configuration
                Highcharts.setOptions({{
                    credits: {{ enabled: false }},
                    chart: {{
                        spacing: [10, 10, 15, 10],
                        margin: [60, 60, 60, 60]
                    }},
                    title: {{
                        margin: 20
                    }},
                    xAxis: {{
                        labels: {{
                            style: {{ fontSize: '12px' }},
                            padding: 5,
                            reserveSpace: true
                        }}
                    }},
                    yAxis: {{
                        labels: {{
                            style: {{ fontSize: '12px' }},
                            x: -5,  // Move y-axis labels left
                            reserveSpace: true
                        }},
                        title: {{
                            margin: 25,  // Increase margin for y-axis title
                            x: -40  // Move y-axis title left
                        }}
                    }},
                    legend: {{
                        margin: 20,  // Add margin to legend
                        y: -15  // Move legend up slightly
                    }},
                    colors: [
                        '#2F80ED', '#27AE60', '#9B51E0', '#F2994A', '#EB5757',
                        '#219653', '#6FCF97', '#56CCF2', '#BB6BD9', '#F2C94C'
                    ]
                }});
                
                // Chart configuration
                {result.code}
            </script>
        </body>
        </html>
        """

        # Create files directory if it doesn't exist
        files_dir = Path("files")
        files_dir.mkdir(exist_ok=True)

        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = f"files/chart_{timestamp}.html"
        save_path = f"files/chart_{timestamp}.png"

        # Save HTML file
        with open(html_path, "w") as f:
            f.write(html_content)

        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1000,800")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Auto-install chromedriver if not present
        try:
            chromedriver_path = chromedriver_autoinstaller.install()
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            logger.warning(f"Error with default Chrome setup: {e}")
            logger.info("Attempting to install/update chromedriver...")
            chromedriver_autoinstaller.install()
            driver = webdriver.Chrome(options=chrome_options)

        try:
            driver.get(f"file://{os.path.abspath(html_path)}")
            time.sleep(2)  # Wait for chart to render
            driver.save_screenshot(save_path)
        finally:
            driver.quit()
            if os.path.exists(html_path):
                os.remove(html_path)

        return save_path, html_path

    def _format_context(self, context: Union[List[str], str]) -> str:
        """Format the context for visualization using LLM."""
        data_extraction_system_prompt = """You are a data extraction expert. Your task is to analyze the given context and format it into a clear, structured text that can be used for visualizations.

Important guidelines:
1. Convert all abbreviated numbers (K, M, B) to their full form
2. Standardize number formats and units
3. Remove any rows/entries with missing or null values
4. Organize related metrics together
5. Present the data in a clear, consistent format
6. Preserve original units (%, $, °C, etc.) and scales
7. Structure the output as a clear text with one data point per line
8. Include relevant metadata (units, time periods, categories) in the description
9. For long contexts with multiple data series:
   - Extract ALL data series, both related and unrelated
   - Group related metrics together under clear headings
   - Separate unrelated data series with clear section breaks
   - Maintain the original relationships between data points
   - Include any contextual information about relationships between series

Example 1:
Input: "Sales by year: 2020: $100K, 2021: $150K, 2022: $200K"
Output:
Sales data in USD:
2020: 100000
2021: 150000
2022: 200000

Example 2:
Input: "Market shares: Company A: 30%, Company B: 25%, Company C: N/A, Others: 25%
Revenue growth: Q1: 5%, Q2: 7%, Q3: 4%
Employee count: Company A: 1000, Company B: 800, Company C: 600"
Output:
Market Share Distribution (excluding Company C due to missing data):
Company A: 30%
Company B: 25%
Others: 25%

Quarterly Revenue Growth:
Q1: 5%
Q2: 7%
Q3: 4%

Employee Distribution:
Company A: 1000
Company B: 800
Company C: 600"""

        data_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=data_extraction_system_prompt),
                ("human", "{context}"),
            ]
        )

        if isinstance(context, list):
            context = "\n".join(str(c) for c in context)

        chain = data_extraction_prompt | self.llm
        formatted_context = chain.invoke({"context": context})

        return formatted_context

    def _run(self, query: str, context: Optional[Union[List[str], str]] = None) -> dict:
        """Run the graph tool."""
        try:
            # Determine if 3D visualization or pie chart is requested
            use_highcharts = (
                "3d" in query.lower()
                or "pie" in query.lower()
                or "piechart" in query.lower()
                or "donut" in query.lower()
                or "doughnut" in query.lower()
            )

            # Format context using LLM if available
            if context:
                formatted_context = self._format_context(context)
                if hasattr(formatted_context, "content"):
                    formatted_context = formatted_context.content
                context = [formatted_context]

            # Prepare chain input
            chain_input = {"query": query}
            if context:
                context_str = "\n".join(str(c) for c in context)
                if context_str.strip():
                    chain_input["context"] = [SystemMessage(content=context_str)]

            # Select appropriate prompt and generate visualization
            prompt = self.highcharts_prompt if use_highcharts else self.altair_prompt
            chart_extractor = prompt | self.llm.with_structured_output(ChartCode)
            result = chart_extractor.invoke(chain_input)

            if use_highcharts:
                # Generate Highcharts visualization
                save_path, _ = self._generate_highcharts_visualization(result)
            else:
                # Generate Altair visualization
                exec_namespace = {"alt": alt, "pd": pd}
                exec(result.code, exec_namespace)

                # Create files directory if it doesn't exist
                files_dir = Path("files")
                files_dir.mkdir(exist_ok=True)

                # Generate unique filename based on timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"files/chart_{timestamp}.png"

                # Save the chart as a PNG file
                exec_namespace["chart"].save(save_path, scale_factor=2.0)

            if os.path.exists(save_path):
                logger.info(f"Image successfully saved to {save_path}")
            else:
                logger.error(
                    f"Image file not found after saving attempt at {save_path}"
                )

            return {
                "message": f"""
                Visualization generated successfully.
                Image saved to: {save_path}
                Reasoning: {result.reasoning}

                Generated Code:
                {result.code}
                """,
                "metadata": {"image_path": save_path},
            }

        except Exception as e:
            logger.error(f"Error in graph tool: {str(e)}")
            return {
                "message": f"Error generating visualization: {str(e)}",
                "metadata": {},
            }

    async def _arun(
        self, query: str, context: Optional[Union[List[str], str]] = None
    ) -> str:
        """Async implementation of the tool."""
        return self._run(query, context)


# Example usage:
if __name__ == "__main__":
    import sys

    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from config import AgentsConfig as Config
    from langchain_openai import ChatOpenAI

    # Ensure the OpenAI API key is set in the environment
    api_key = Config().OPENAI_API_KEY
    # Instantiate the ChatOpenAI model
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0,  # Use 0 for deterministic outputs in math calculations
    )

    # Create tool
    graph_tool = GraphTool(llm=llm)

    # Test the tool
    test_query = "Show me sales growth over time"
    test_context = ["Sales by year: 2020: $100K, 2021: $130K, 2022: $200K"]

    result = graph_tool._run(test_query, test_context)
    print(result)
