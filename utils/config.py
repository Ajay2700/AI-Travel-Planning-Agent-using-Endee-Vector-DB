"""
Centralized configuration for the AI Travel Planning Agent.

All configuration is loaded from environment variables (via .env file or system env).
This makes it easy to configure for different environments without changing code.
"""

import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
# This allows users to configure the app without modifying code
load_dotenv()

# OpenAI / LLM configuration
# These are optional - the app works in fallback mode without them
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # Default to a cost-effective model

# Embedding model
# all-MiniLM-L6-v2 is a good balance of quality and speed (384 dimensions)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Endee vector database configuration
# Default assumes Endee is running locally on port 8080
ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "travel_plans")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")  # Optional authentication token

# General settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))  # Default number of results to retrieve


def _init_logger() -> logging.Logger:
    """Configure and return a root logger for the app."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("ai_travel_agent")


logger = _init_logger()

