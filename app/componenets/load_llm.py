from typing import Optional
from langchain_groq import ChatGroq
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException    
from app.config.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = get_logger(__name__)


def load_llm(
    model_name: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Optional[ChatGroq]:
    """
    Initialize and return a ChatGroq LLM instance using configured model and credentials.
    """
    key = groq_api_key or GROQ_API_KEY
    model = model_name or LLM_MODEL
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    if not key:
        logger.error("GROQ_API_KEY is not set. Please provide a valid GROQ_API_KEY in your .env file.")
        return None

    try:
        logger.info(f"Initializing LLM model '{model}' with temperature={temp}, max_tokens={tokens}...")
        llm = ChatGroq(
            model_name=model,
            groq_api_key=key,
            temperature=temp,
            max_tokens=tokens
        )
        logger.info(f"LLM model '{model}' initialized successfully.")
        return llm
    except Exception as e:
        error_msg = CustomException(f"Failed to load LLM model '{model}'", e)
        logger.error(str(error_msg)) 
        return None
