# Import required modules
from dataclasses import dataclass, field
from typing import Dict, List
import threading
from datetime import datetime

# Dataclass to store token usage information for each API call
@dataclass
class TokenUsage:
    """
    Stores information about token usage and cost for a single API call.

    Args:
        input_tokens (int): Number of input tokens used
        output_tokens (int): Number of output tokens generated
        cost (float): Cost in USD for this API call
        timestamp (str): ISO format timestamp of when the usage occurred
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# Singleton class to track API usage costs across the application
class CostTracker:
    """
    A thread-safe singleton class that tracks token usage and costs across all API calls.
    Maintains running totals and historical usage data.
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

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float):
        """
        Records a new API usage instance.

        Args:
            input_tokens (int): Number of input tokens used
            output_tokens (int): Number of output tokens generated
            cost (float): Cost in USD for this API call
        """
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
        self.usage_data.append(usage)
        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_metrics(self) -> Dict:
        """
        Returns current usage metrics.

        Returns:
            Dict: Dictionary containing total cost, token counts, and number of API calls
        """
        return {
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "calls": len(self.usage_data)
        }

    def reset(self):
        """Resets all usage metrics and clears historical data."""
        self.usage_data = []
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0 