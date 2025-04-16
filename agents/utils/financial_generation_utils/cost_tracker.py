# Import required modules
from dataclasses import dataclass, field
from typing import Dict, List
import threading
from datetime import datetime


# Dataclass to store token usage information for a single API call
@dataclass
class TokenUsage:
    """
    A dataclass to track token usage metrics for individual API calls.

    Args:
        input_tokens (int): Number of input tokens used
        output_tokens (int): Number of output tokens generated
        cost (float): Cost in USD for this API call
        timestamp (str): ISO format timestamp of the API call
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Singleton class to track overall API usage and costs
class CostTracker:
    """
    A singleton class that tracks and aggregates token usage and costs across API calls.
    Provides methods to add usage data, retrieve metrics, and reset tracking.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.usage_data = []
                    cls._instance.total_cost = 0.0
                    cls._instance.total_input_tokens = 0
                    cls._instance.total_output_tokens = 0
        return cls._instance

    # Add usage data for a single API call
    # Args: input_tokens (int), output_tokens (int), cost (float)
    def add_usage(self, input_tokens: int, output_tokens: int, cost: float):
        usage = TokenUsage(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )
        self.usage_data.append(usage)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    # Get aggregated metrics for all tracked API calls
    # Returns: Dict containing total cost, tokens, and number of calls
    def get_metrics(self) -> Dict:
        return {
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "calls": len(self.usage_data),
        }

    # Reset all tracking metrics to initial values
    def reset(self):
        self.usage_data = []
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
