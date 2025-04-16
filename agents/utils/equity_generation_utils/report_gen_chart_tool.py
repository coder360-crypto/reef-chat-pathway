# Import required libraries for file handling and environment variables
import os
import openai
import sys
import logging
from dotenv import load_dotenv

# Import libraries for stock data and visualization
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.equity_generation_utils.financial_metrics import get_stock_ticker
from langchain_openai import ChatOpenAI
from plotly.subplots import make_subplots

# Configure logging settings
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    A class to generate stock analysis reports and charts.
    
    Methods:
        generate_stock_chart(company_name: str) -> bytes
            Generates a stock price chart with volume data for the specified company
    """
    def __init__(self):
        self._setup_logging()
        self._load_env()

    def _setup_logging(self):
        self.logger = logger

    def _load_env(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # openai.api_base = os.getenv('GROQ_URL')
        self.logger.info("Loaded environment variables")

    def generate_stock_chart(self, company_name: str) -> str:
        """Generate stock price trend chart for the last 6 months and return the path to saved PNG
        
        Args:
            company_name (str): Name of the company to generate chart for
            
        Returns:
            bytes: PNG image data of the generated chart
        """
        try:
            ticker_symbol = get_stock_ticker(company_name)
            ticker = yf.Ticker(ticker_symbol)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)

            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No historical data available for {ticker_symbol}")

            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
            )

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    increasing_line_color="#00873c",
                    decreasing_line_color="#cf1124",
                    increasing_line_width=3,
                    decreasing_line_width=3,
                    name="Price",
                ),
                row=1,
                col=1,
            )

            # Add volume bar chart
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["Volume"],
                    name="Volume",
                    marker_color="rgba(0,0,150,0.5)",
                ),
                row=2,
                col=1,
            )

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"{company_name} ({ticker_symbol}) Stock Analysis",
                    font=dict(size=24, color="black"),
                ),
                template="plotly_white",
                xaxis_rangeslider_visible=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="black"),
                showlegend=True,
                height=900,
                yaxis=dict(
                    title="Price (USD)",
                    showgrid=False,
                    tickfont=dict(size=12, color="black"),
                    showline=True,
                    linewidth=1,
                    linecolor="lightgrey",
                ),
                yaxis2=dict(
                    title="Volume",
                    showgrid=False,
                    tickfont=dict(size=12, color="black"),
                    showline=True,
                    linewidth=1,
                    linecolor="lightgrey",
                ),
                margin=dict(l=50, r=50, t=80, b=50),
            )

            # Update axes - remove grid, add subtle axis lines
            fig.update_xaxes(
                showgrid=False,
                tickfont=dict(size=12, color="black"),
                showline=True,
                linewidth=1,
                linecolor="lightgrey",
            )
            fig.update_yaxes(
                showgrid=False,
                tickfont=dict(size=12, color="black"),
                showline=True,
                linewidth=1,
                linecolor="lightgrey",
            )

            # Convert figure to bytes array
            img_bytes = fig.to_image(format="png", width=1600, height=900, scale=3)

            return img_bytes

        except Exception as e:
            self.logger.error(f"Error generating stock chart: {str(e)}")
            return None


if __name__ == "__main__":
    generator = ReportGenerator()
    chart_path = generator.generate_stock_chart("Tesla")
    if chart_path:
        print(f"Generated chart at: {chart_path}")
