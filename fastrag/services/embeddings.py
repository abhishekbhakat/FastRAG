from typing import Any

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def get_openai_embedding(config: dict[str, Any]) -> OpenAIEmbedding:
    return OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])



def get_ollama_embedding(config: dict[str, Any]) -> OllamaEmbedding:
    return OllamaEmbedding(model_name=config["embedding_model"])


def get_lmstudio_embedding(config: dict[str, Any]) -> OpenAIEmbedding:
    return OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])


def get_embedding_model(config: dict[str, Any]) -> OpenAIEmbedding | OllamaEmbedding:
    embedding_provider = config["embeddings_provider"]
    if embedding_provider == "openai":
        return get_openai_embedding(config)
    elif embedding_provider == "ollama":
        return get_ollama_embedding(config)
    elif embedding_provider == "lmstudio":
        return get_lmstudio_embedding(config)
    else:
        raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")
