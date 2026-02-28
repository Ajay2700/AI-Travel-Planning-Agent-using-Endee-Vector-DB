import logging
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI / LLM configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Endee vector database configuration
ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8080")
ENDEE_INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "travel_plans")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "")  # Optional authentication

# General settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))


def _init_logger() -> logging.Logger:
    """Configure and return a root logger for the app."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("ai_travel_agent")


logger = _init_logger()

