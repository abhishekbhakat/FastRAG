import os

from fastrag.config import config, logger
from fastrag.services.storage_context import create_index, get_index, set_storage_context
from fastrag.services.vector_store import get_vector_store


def new_storage_context():
    # Initialize vector store
    vs = get_vector_store()

    # Initialize storage context and create a new index if the existing one cannot be loaded
    sc = set_storage_context(vs)
    index, sc = create_index(documents=[], vector_store=vs)
    config.update({"index": index, "vector_store": vs, "storage_context": sc})


def bootstrap_app():
    logger.info("Bootstrapping application")

    # create cache dir if it doesn't exist
    if not os.path.exists(config["cache_dir"]):
        os.makedirs(config["cache_dir"])
        new_storage_context()
        logger.info("Application bootstrapped successfully")
        return

    # Create upload directory if it doesn't exist
    if not os.path.exists(config["upload_dir"]):
        os.makedirs(config["upload_dir"])
        logger.info("Upload directory created successfully")

    # Initialize vector store
    vs = get_vector_store()

    # Attempt to load the existing index
    try:
        index, sc = get_index(vector_store=vs)
        if index:
            logger.info("Existing index loaded successfully")
            config.update({"index": index, "vector_store": vs, "storage_context": sc})
    except Exception as e:
        logger.warning(f"Failed to load existing index: {e}")
        new_storage_context()

    logger.info("Application bootstrapped successfully")
    return
