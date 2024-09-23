from pathlib import Path
from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

from fastrag.config import logger
from fastrag.services.embeddings import get_embedding_model


def set_storage_context(vector_store: PGVectorStore, cache_dir: str | None = None) -> StorageContext:
    logger.info("Setting up storage context")
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    storage_context = StorageContext.from_defaults(docstore=docstore, index_store=index_store, vector_store=vector_store)
    if cache_dir:
        logger.debug(f"Persisting storage context to cache directory: {cache_dir}")
        storage_context.persist(persist_dir=cache_dir)
    return storage_context


def persist_storage_context(storage_context: StorageContext, cache_dir: str | None = None):
    if cache_dir:
        logger.info(f"Persisting storage context to cache directory: {cache_dir}")
        cache_path = Path(cache_dir)
        storage_context.persist(persist_dir=cache_path)


def get_storage_context(vector_store: PGVectorStore, cache_dir: str | None = None) -> StorageContext:
    logger.info("Getting storage context")
    if cache_dir:
        logger.debug(f"Loading storage context from cache directory: {cache_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=cache_dir, vector_store=vector_store)
    else:
        logger.debug("Creating new storage context")
        storage_context = set_storage_context(vector_store, cache_dir)
    return storage_context


def create_index(config: dict[str, Any], documents: list[Document], vector_store: PGVectorStore, cache_dir: str | None = None) -> VectorStoreIndex:
    logger.info("Creating index")
    # TODO: Instead of config it should expect embedding model and vector store
    embed_model = get_embedding_model(config=config)
    storage_context = get_storage_context(vector_store, cache_dir)
    logger.debug(f"Creating index with {len(documents)} documents")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, embed_batch_size=10, index_batch_size=100)
    logger.info("Index created successfully")
    persist_storage_context(storage_context, cache_dir)
    return index, storage_context


def get_index(config: dict[str, Any], vector_store: PGVectorStore, cache_dir: str | None = None) -> tuple[VectorStoreIndex, StorageContext]:
    logger.info("Getting index")
    if cache_dir:
        try:
            logger.debug(f"Loading index from cache directory: {cache_dir}")
            storage_context = get_storage_context(vector_store=vector_store, cache_dir=cache_dir)
            index = load_index_from_storage(storage_context, embed_model=get_embedding_model(config=config))
            logger.info("Index loaded successfully")
            return index, storage_context
        except ValueError:
            logger.warning("Index not found in cache directory, returning none for index")
            return None, storage_context
    else:
        logger.warning("No cache directory provided, returning None for index and storage context")
        return None, None
