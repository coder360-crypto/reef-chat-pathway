# Import required libraries for API health checking and email notifications
import logging
import os
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from config import AgentsConfig as Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()


class APIHealthChecker:
    """
    A class to monitor and check the health of various API endpoints.
    
    Args:
        smtp_server (str): SMTP server address for sending email alerts
        smtp_port (int): SMTP server port number
        sender_email (str): Email address to send alerts from
        sender_password (str): Password for sender email account
        recipient_email (str): Email address to receive alerts
    """
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email

        # Define API endpoints with comprehensive configurations
        self.endpoints = [
            {
                "name": "OpenAI API",
                "method": "POST",
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                "payload": {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
                "expected_status": 200,
            },
            {
                "name": "Serper API",
                "method": "POST",
                "url": "https://google.serper.dev/search",
                "headers": {
                    "X-API-KEY": config.SERPER_API_KEY,
                    "Content-Type": "application/json",
                },
                "payload": {"q": "test query", "num": 1},
                "expected_status": 200,
            },
            {
                "name": "Groq API",
                "method": "POST",
                "url": "https://api.groq.com/openai/v1/chat/completions",
                "headers": {
                    "Authorization": f"Bearer {config.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                "payload": {
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
                "expected_status": 200,
            },
            {
                "name": "Cohere API",
                "method": "POST",
                "url": "https://api.cohere.ai/v1/generate",
                "headers": {
                    "Authorization": f"Bearer {config.COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                "payload": {
                    "prompt": "Hello",
                    "max_tokens": 5,
                },
                "expected_status": 200,
            },
            # {
            #     "name": "Tavily API",
            #     "method": "POST",
            #     "url": "https://api.tavily.com/search",
            #     "headers": {
            #         "api-key": config.TAVILY_API_KEY,
            #     },
            #     "payload": {
            #         "query": "test",
            #         "api_key": config.TAVILY_API_KEY,
            #         "max_results": 1,
            #     },
            #     "expected_status": 200,
            # },
            {
                "name": "Alpha Vantage API",
                "method": "GET",
                "url": "https://www.alphavantage.co/query",
                "params": {
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": "IBM",
                    "interval": "5min",
                    "apikey": config.ALPHAVANTAGE_API_KEY,
                },
                "expected_status": 200,
            },
            {
                "name": "Polygon API",
                "method": "GET",
                "url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09",
                "params": {
                    "apiKey": config.POLYGON_API_KEY,
                },
                "expected_status": 200,
            },
            {
                "name": "Finnhub API",
                "method": "GET",
                "url": "https://finnhub.io/api/v1/quote",
                "params": {
                    "symbol": "AAPL",
                    "token": config.FINHUB_API_KEY,
                },
                "expected_status": 200,
            },
            {
                "name": "CourtListener API",
                "method": "GET",
                "url": "https://www.courtlistener.com/api/rest/v4/dockets/",
                "headers": {
                    "Authorization": f"Token {config.COURTLISTENER_API_KEY}",
                },
                "expected_status": 200,
            },
            {
                "name": "Financial Datasets",
                "method": "GET",
                "url": "https://api.financialdatasets.ai/financials/income-statements",
                "headers": {
                    "X-API-KEY": config.FINANCIAL_DATASETS_API_KEY,
                    "Content-Type": "application/json",
                },
                "params": {"ticker": "NVDA", "limit": 5, "period": "annual"},
                "expected_status": 200,
            },
            {
                "name": "Jina Search",
                "method": "GET",
                "url": "https://s.jina.ai/https://www.google.com",
                "headers": {"Authorization": f"Bearer {config.JINA_API_KEY}"},
                "expected_status": 200,
                "timeout": 30,
                "retries": 2,
            },
        ]

    def check_endpoint(self, endpoint: Dict) -> Dict:
        """
        Check the health of a single API endpoint.
        
        Args:
            endpoint (Dict): Dictionary containing endpoint configuration
            
        Returns:
            Dict: Health check results including status and error details
        """
        retries = endpoint.get("retries", 0)
        timeout = endpoint.get("timeout", 10)

        for attempt in range(retries + 1):
            try:
                kwargs = {
                    "method": endpoint["method"],
                    "url": endpoint["url"],
                    "timeout": timeout,
                }

                # Add headers if present
                if "headers" in endpoint:
                    kwargs["headers"] = endpoint["headers"]

                # Add payload for POST requests
                if "payload" in endpoint:
                    kwargs["json"] = endpoint["payload"]

                # Add URL parameters for GET requests
                if "params" in endpoint:
                    kwargs["params"] = endpoint["params"]

                response = requests.request(**kwargs)

                # Check if response is JSON
                try:
                    response_data = response.json()
                except:
                    response_data = None

                is_healthy = response.status_code == endpoint["expected_status"]
                error_message = (
                    None if is_healthy else f"Status code: {response.status_code}"
                )

                # Additional error checking for specific APIs
                if is_healthy and response_data:
                    if "error" in response_data:
                        is_healthy = False
                        error_message = f"API Error: {response_data['error']}"
                    elif (
                        endpoint["name"] == "OpenAI API"
                        and "choices" not in response_data
                    ):
                        is_healthy = False
                        error_message = "Invalid OpenAI API response format"

                if error_message:
                    logger.error(f"Error for {endpoint['name']}: {error_message}")

                return {
                    "name": endpoint["name"],
                    "healthy": is_healthy,
                    "status_code": response.status_code,
                    "error": error_message,
                    "response": response_data if not is_healthy else None,
                }

            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {endpoint['name']}, retrying..."
                    )
                    continue
                logger.error(f"Request failed for {endpoint['name']}: {str(e)}")
                return {
                    "name": endpoint["name"],
                    "healthy": False,
                    "status_code": None,
                    "error": f"Request failed: {str(e)}",
                    "response": None,
                }
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint['name']}: {str(e)}")
                return {
                    "name": endpoint["name"],
                    "healthy": False,
                    "status_code": None,
                    "error": f"Unexpected error: {str(e)}",
                    "response": None,
                }

    def send_email_alert(self, failed_endpoints: List[Dict]):
        """
        Send email notification for failed API endpoints.
        
        Args:
            failed_endpoints (List[Dict]): List of endpoints that failed health check
        """
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email
        msg["Subject"] = (
            f"API Health Check Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        body = "The following APIs are experiencing issues:\n\n"
        for endpoint in failed_endpoints:
            body += f"API: {endpoint['name']}\n"
            body += f"Status Code: {endpoint['status_code']}\n"
            body += f"Error: {endpoint['error']}\n\n"

        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                logger.info("Alert email sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def run_health_check(self):
        """
        Run health checks on all configured API endpoints and send alerts if needed.
        """
        logger.info("Starting API health check")
        failed_endpoints = []

        for endpoint in self.endpoints:
            result = self.check_endpoint(endpoint)
            if not result["healthy"]:
                failed_endpoints.append(result)
                logger.warning(
                    f"API check failed for {result['name']}: {result['error']}"
                )
            else:
                logger.info(
                    f"{result['name']} is healthy. Response: {result['response']}"
                )

        if failed_endpoints:
            self.send_email_alert(failed_endpoints)
        else:
            logger.info("All APIs are healthy")


if __name__ == "__main__":
    # Initialize health checker with email configuration
    checker = APIHealthChecker(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email=config.EMAIL_ADDRESS,
        sender_password=config.EMAIL_PASSWORD,
        recipient_email=config.EMAIL_ADDRESS,
    )

    # Run the health check
    checker.run_health_check()
