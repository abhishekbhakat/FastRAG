import os

from fastrag.config import config, logger
from fastrag.services.storage_context import create_index, get_index, set_storage_context
from fastrag.services.vector_store import get_vector_store


def bootstrap_app():
    logger.info("Bootstrapping application")

    # create cache dir if it doesn't exist
    if not os.path.exists(config["cache_dir"]):
        os.makedirs(config["cache_dir"])

    # Initialize vector store
    vs = get_vector_store(config)

    # Attempt to load the existing index
    index, _ = get_index(config=config, vector_store=vs, cache_dir=config["cache_dir"])
    if index:
        logger.info("Existing index loaded successfully")
        return

    # Initialize storage context and create a new index if the existing one cannot be loaded
    _ = set_storage_context(vs, config["cache_dir"])
    index = create_index(config=config, documents=[], vector_store=vs, cache_dir=config["cache_dir"])

    logger.info("Application bootstrapped successfully")
    return
