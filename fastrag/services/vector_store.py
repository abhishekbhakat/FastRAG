from typing import Any
from urllib.parse import urlparse

from llama_index.vector_stores.postgres import PGVectorStore

from fastrag.config import logger


def get_vector_store(config: dict[str, Any]) -> PGVectorStore:
    logger.info("Initializing vector store")
    try:
        # Parse the database URL
        url = urlparse(config["database_url"])
        logger.debug(f"Parsed database URL: {url.scheme}://{url.hostname}:{url.port}{url.path}")

        # Extract connection details
        username = url.username
        password = url.password
        host = url.hostname
        port = url.port
        database = url.path[1:]  # Remove the leading '/'

        # Create and return the PGVectorStore
        vector_store = PGVectorStore.from_params(
            database=database,
            host=host,
            password=password,
            port=port,
            user=username,
            table_name="vector_store",
            embed_dim=config["embed_dimension"],
        )
        logger.info("Vector store initialized successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        raise
