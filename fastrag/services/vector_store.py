from urllib.parse import urlparse

from llama_index.vector_stores.postgres import PGVectorStore

from fastrag.config import config, logger
from fastrag.models import ServiceStatus


def get_vector_store() -> PGVectorStore:
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
        save_vector_store_status("online")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        save_vector_store_status("offline")
        raise


def save_vector_store_status(status: str):
    from datetime import datetime

    ServiceStatus(service_name="VectorStore", status=status, last_checked=datetime.now().isoformat()).save()
