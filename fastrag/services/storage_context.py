from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.postgres import PGVectorStore

from fastrag.config import config, logger
from fastrag.services.embeddings import get_embedding_model

cache_dir = config["cache_dir"]


def set_storage_context(vector_store: PGVectorStore) -> StorageContext:
    logger.info("Setting up storage context")
    docstore = SimpleDocumentStore()
    index_store = SimpleIndexStore()
    storage_context = StorageContext.from_defaults(docstore=docstore, index_store=index_store, vector_store=vector_store)
    if cache_dir:
        logger.debug(f"Persisting storage context to cache directory: {cache_dir}")
        storage_context.persist(persist_dir=cache_dir)
    return storage_context


def persist_storage_context(storage_context: StorageContext):
    if cache_dir:
        logger.info(f"Persisting storage context to cache directory: {cache_dir}")
        cache_path = Path(cache_dir)
        storage_context.persist(persist_dir=cache_path)


def get_storage_context(vector_store: PGVectorStore) -> StorageContext:
    logger.info("Getting storage context")
    if cache_dir:
        logger.debug(f"Loading storage context from cache directory: {cache_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=cache_dir, vector_store=vector_store)
    else:
        logger.debug("Creating new storage context")
        storage_context = set_storage_context(vector_store, cache_dir)
    return storage_context


def create_index(documents: list[Document], vector_store: PGVectorStore) -> tuple[VectorStoreIndex | None, StorageContext | None]:
    logger.info("Creating index")
    # TODO: Instead of config it should expect embedding model and vector store
    embed_model = get_embedding_model(config=config)
    storage_context = get_storage_context(vector_store, cache_dir)
    logger.debug(f"Creating index with {len(documents)} documents")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, embed_batch_size=10, index_batch_size=100)
    logger.info("Index created successfully")
    persist_storage_context(storage_context, cache_dir)
    return index, storage_context


def get_index(vector_store: PGVectorStore) -> tuple[VectorStoreIndex | None, StorageContext | None]:
    logger.info("Getting index")
    storage_context = None  # Initialize storage_context to None
    if cache_dir:
        try:
            logger.debug(f"Loading index from cache directory: {cache_dir}")
            storage_context = get_storage_context(vector_store=vector_store)
            index = load_index_from_storage(storage_context, embed_model=get_embedding_model())
            logger.info("Index loaded successfully")
            return index, storage_context
        except ValueError:
            logger.warning("Index not found in cache directory, returning none for index")
            return None, storage_context
    else:
        logger.warning("No cache directory provided, returning None for index and storage context")
        return None, None


def get_all_documents(index: VectorStoreIndex) -> list[dict]:
    logger.info("Getting all documents")
    documents = []
    for ref_doc_id, doc_info in index.ref_doc_info.items():
        doc_metadata = doc_info.metadata.copy()
        doc_metadata["ref_doc_id"] = ref_doc_id
        documents.append(doc_metadata)
    return documents


def get_ref_doc_id(index: VectorStoreIndex, metadata: dict) -> str | None:
    for ref_doc_id, doc_info in index.ref_doc_info.items():
        if all(doc_info.metadata.get(k) == v for k, v in metadata.items()):
            return ref_doc_id
    return None


def get_document(index: VectorStoreIndex, storage_context: StorageContext, metadata: dict) -> dict | None:
    ref_doc_id = get_ref_doc_id(metadata)
    if ref_doc_id:
        doc_info = index.ref_doc_info[ref_doc_id]
        return {"ref_doc_id": ref_doc_id, "metadata": doc_info.metadata, "text": storage_context.docstore.get_document(ref_doc_id).text}
    return None


def delete_document(index: VectorStoreIndex, storage_context: StorageContext, metadata: dict) -> bool:
    ref_doc_id = get_ref_doc_id(metadata)
    if ref_doc_id:
        index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
        storage_context.persist(persist_dir=cache_dir)
        return True
    return False
