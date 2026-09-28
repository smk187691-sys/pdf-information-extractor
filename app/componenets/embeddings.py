from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import EMBEDDING_MODEL

logger = get_logger(__name__)

_cached_embedding_model: Optional[HuggingFaceEmbeddings] = None
_cached_model_name: Optional[str] = None


def get_embedding_model(model_name: Optional[str] = None) -> Optional[HuggingFaceEmbeddings]:
    """
    Get or initialize a cached HuggingFace embeddings model.
    Reuses existing instance to avoid redundant model loads.
    """
    global _cached_embedding_model, _cached_model_name
    target_model = model_name or EMBEDDING_MODEL

    if _cached_embedding_model is not None and _cached_model_name == target_model:
        return _cached_embedding_model

    try:
        logger.info(f"Initializing HuggingFace Embeddings model with: '{target_model}'...")
        model = HuggingFaceEmbeddings(
            model_name=target_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        _cached_embedding_model = model
        _cached_model_name = target_model
        logger.info(f"HuggingFace Embeddings model '{target_model}' initialized successfully.")
        return model
    except Exception as e:
        error_msg = CustomException(f"Failed to create embeddings with model '{target_model}'.", e)
        logger.error(str(error_msg))
        return None
