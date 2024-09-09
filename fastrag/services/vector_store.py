from typing import Any
from urllib.parse import urlparse

from llama_index.vector_stores.postgres import PGVectorStore


def get_vector_store(config: dict[str, Any]) -> PGVectorStore:
    # Parse the database URL
    url = urlparse(config["database_url"])

    # Extract connection details
    username = url.username
    password = url.password
    host = url.hostname
    port = url.port
    database = url.path[1:]  # Remove the leading '/'

    # Create and return the PGVectorStore
    return PGVectorStore.from_params(
        database=database,
        host=host,
        password=password,
        port=port,
        user=username,
        table_name="vector_store",
        embed_dim=config["embed_dimension"],
    )
