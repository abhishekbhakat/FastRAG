from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

from fastrag.services.embeddings import get_embedding_model


def set_storage_context(vector_store: PGVectorStore, cache_dir: str | None = None) -> StorageContext:
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    storage_context = StorageContext.from_defaults(docstore=docstore, index_store=index_store, vector_store=vector_store)
    if cache_dir:
        storage_context.persist(persist_dir=cache_dir)
    return storage_context


def persist_storage_context(storage_context: StorageContext, cache_dir: str | None = None):
    if cache_dir:
        storage_context.persist(persist_dir=cache_dir)


def get_storage_context(vector_store: PGVectorStore, cache_dir: str | None = None) -> StorageContext:
    if cache_dir:
        storage_context = StorageContext.from_defaults(persist_dir=cache_dir, vector_store=vector_store)
    else:
        storage_context = set_storage_context(vector_store, cache_dir)
    return storage_context


def create_index(config: dict[str, Any], documents: list[Document], vector_store: PGVectorStore, cache_dir: str | None = None) -> VectorStoreIndex:
    # TODO: Instead of config it should expect embedding model and vector store
    embed_model = get_embedding_model(config=config)
    storage_context = get_storage_context(vector_store, cache_dir)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, embed_batch_size=10, index_batch_size=100)
    persist_storage_context(storage_context, cache_dir)
    return index


def get_index(config: dict[str, Any], vector_store: PGVectorStore, cache_dir: str | None = None) -> tuple[VectorStoreIndex, StorageContext]:
    if cache_dir:
        storage_context = get_storage_context(vector_store=vector_store, cache_dir=cache_dir)
        index = load_index_from_storage(storage_context, embed_model=get_embedding_model(config=config))
        return index, storage_context
    else:
        return None, None
