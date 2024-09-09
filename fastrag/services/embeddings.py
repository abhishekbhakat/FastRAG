from typing import Any

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from fastrag.config import logger


def get_openai_embedding(config: dict[str, Any]) -> OpenAIEmbedding:
    logger.info("Initializing OpenAI embedding")
    return OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])


def get_ollama_embedding(config: dict[str, Any]) -> OllamaEmbedding:
    logger.info("Initializing Ollama embedding")
    return OllamaEmbedding(model_name=config["embedding_model"])


def get_lmstudio_embedding(config: dict[str, Any]) -> OpenAIEmbedding:
    logger.info("Initializing LMStudio embedding")
    return OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])


def get_embedding_model(config: dict[str, Any]) -> OpenAIEmbedding | OllamaEmbedding:
    logger.info("Getting embedding model")
    embedding_provider = config["embeddings_provider"]
    logger.debug(f"Embedding provider: {embedding_provider}")
    try:
        if embedding_provider == "openai":
            return get_openai_embedding(config)
        elif embedding_provider == "ollama":
            return get_ollama_embedding(config)
        elif embedding_provider == "lmstudio":
            return get_lmstudio_embedding(config)
        else:
            logger.error(f"Unsupported embeddings provider: {embedding_provider}")
            raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        raise
