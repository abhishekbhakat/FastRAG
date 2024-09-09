import os

from fastrag.config import config, logger
from fastrag.services.storage_context import set_storage_context
from fastrag.services.vector_store import get_vector_store


def bootstrap_app():
    logger.info("Bootstrapping application")

    # create cache dir if it doesn't exist
    if not os.path.exists(config["cache_dir"]):
        os.makedirs(config["cache_dir"])

    # Initialize vector store
    vs = get_vector_store(config)

    # Initialize storage context and index
    _ = set_storage_context(vs, config["cache_dir"])

    logger.info("Application bootstrapped successfully")
    return
