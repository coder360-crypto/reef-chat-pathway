# Import required libraries
import os
import sys
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from nemoguardrails import RailsConfig, LLMRails

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import re


class GuardrailManager:
    """
    Manages the guardrails configuration and validation for LLM responses.
    
    Args:
        None
        
    Attributes:
        config (RailsConfig): Configuration object for the guardrails
        rails (LLMRails): LLM guardrails instance
    """
    def __init__(self):
        config_path = "./config/config.yml"
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        updated_content = re.sub(
            r'api_key:.*',
            f'api_key: "{api_key}"',
            config_content
        )

        print(updated_content)
        
        # Write updated config with API key
        with open(config_path, 'w') as f:
            f.write(updated_content)

        # Initialize rails configuration
        self.config = RailsConfig.from_path("./config")
        self.rails = LLMRails(self.config)

        # Restore original config content
        with open(config_path, 'w') as f:
            f.write(config_content)

    # Args: query (str): Input text to validate
    # Returns: dict: Contains validation verdict and guarded query
    def validate(self, query: str):
        response = self.rails.generate(messages=[{
            "role": "user",
            "content":query
        }]) 

        info = self.rails.explain()
        is_Valid = "no" in info.llm_calls[-1].completion.lower()
        return {"verdict": is_Valid, "guarded_query": response["content"]}
    

# Pydantic model for validation request
class ValidationRequest(BaseModel):
    query: str


# Initialize guardrail manager instance
guardrail_manager = GuardrailManager()

# Args: query (str): Input text to validate
# Returns: dict: Validation results from guardrail manager
def validate_input(query: str):
    return guardrail_manager.validate(query)

# Initialize FastAPI app
app = FastAPI()

# FastAPI endpoint for validation
@app.post("/validate")
def validate_endpoint(request: ValidationRequest):
    return validate_input(request.query)


# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
