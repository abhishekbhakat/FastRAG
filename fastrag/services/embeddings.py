from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from fastrag.config import config, logger
from fastrag.models import ServiceStatus


def get_openai_embedding() -> OpenAIEmbedding:
    logger.info("Initializing OpenAI embedding")
    try:
        embedding = OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])
        save_embedding_status("online")
        return embedding
    except Exception as e:
        logger.error(f"Error initializing OpenAI embedding: {str(e)}", exc_info=True)
        save_embedding_status("offline")
        raise


def get_ollama_embedding() -> OllamaEmbedding:
    logger.info("Initializing Ollama embedding")
    try:
        embedding = OllamaEmbedding(model_name=config["embedding_model"])
        save_embedding_status("online")
        return embedding
    except Exception as e:
        logger.error(f"Error initializing Ollama embedding: {str(e)}", exc_info=True)
        save_embedding_status("offline")
        raise


def get_lmstudio_embedding() -> OpenAIEmbedding:
    logger.info("Initializing LMStudio embedding")
    try:
        embedding = OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])
        save_embedding_status("online")
        return embedding
    except Exception as e:
        logger.error(f"Error initializing LMStudio embedding: {str(e)}", exc_info=True)
        save_embedding_status("offline")
        raise


def get_embedding_model() -> OpenAIEmbedding | OllamaEmbedding:
    logger.info("Getting embedding model")
    embedding_provider = config["embeddings_provider"]
    logger.debug(f"Embedding provider: {embedding_provider}")
    try:
        if embedding_provider == "openai":
            return get_openai_embedding()
        elif embedding_provider == "ollama":
            return get_ollama_embedding()
        elif embedding_provider == "lmstudio":
            return get_lmstudio_embedding()
        else:
            logger.error(f"Unsupported embeddings provider: {embedding_provider}")
            raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        raise


def save_embedding_status(status: str):
    from datetime import datetime

    ServiceStatus(service_name="Embedding", status=status, last_checked=datetime.now().isoformat()).save()
