from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.types import MetadataFilters
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
        storage_context = set_storage_context(vector_store)
    return storage_context


def create_index(documents: list[Document], vector_store: PGVectorStore) -> tuple[VectorStoreIndex | None, StorageContext | None]:
    logger.info("Creating index")
    # TODO: Instead of config it should expect embedding model and vector store
    embed_model = get_embedding_model()
    storage_context = get_storage_context(vector_store)
    logger.debug(f"Creating index with {len(documents)} documents")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, embed_batch_size=10, index_batch_size=100)
    logger.info("Index created successfully")
    persist_storage_context(storage_context)
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
    try:
        vector_store = index._vector_store
        logger.debug(f"Vector store: {vector_store}")
        logger.debug(f"Vector store type: {type(vector_store)}")

        if isinstance(vector_store, PGVectorStore):
            # Create a dummy filter that always evaluates to true
            dummy_filter = MetadataFilters(filters=[MetadataFilters(filters=[], condition="or")], condition="and")

            nodes = vector_store.get_nodes(filters=dummy_filter)

            logger.debug(f"Retrieved {len(nodes)} documents from vector store")

            documents = []
            for node in nodes:
                doc_metadata = node.metadata.copy() if node.metadata else {}
                doc_metadata["ref_doc_id"] = node.node_id
                doc_metadata["text"] = node.get_content()
                documents.append(doc_metadata)

            logger.info(f"Processed {len(documents)} documents")
            return documents
        else:
            logger.warning(f"Unsupported vector store type: {type(vector_store)}")
            return []
    except Exception as e:
        logger.error(f"Error getting all documents: {str(e)}", exc_info=True)
        return []


def get_ref_doc_id(index: VectorStoreIndex, metadata: dict) -> str | None:
    vector_store = index._vector_store
    if isinstance(vector_store, PGVectorStore):
        # Construct a query to find a document with matching metadata
        query = "SELECT id FROM vector_store WHERE "
        conditions = []
        for key, value in metadata.items():
            conditions.append(f"metadata->'{key}' = '{value}'")
        query += " AND ".join(conditions)

        result = vector_store.client.query(query)
        if result:
            return result[0]["id"]
    return None


def get_document(index: VectorStoreIndex, storage_context: StorageContext, metadata: dict) -> dict | None:
    ref_doc_id = get_ref_doc_id(index, metadata)
    if ref_doc_id:
        vector_store = index._vector_store
        if isinstance(vector_store, PGVectorStore):
            result = vector_store.client.query(f"SELECT * FROM vector_store WHERE id = '{ref_doc_id}'")
            if result:
                doc = result[0]
                return {"ref_doc_id": ref_doc_id, "metadata": doc.get("metadata", {}), "text": doc.get("text", "")}
    return None


def delete_document(index: VectorStoreIndex, storage_context: StorageContext, metadata: dict) -> bool:
    ref_doc_id = get_ref_doc_id(index, metadata)
    if ref_doc_id:
        vector_store = index._vector_store
        if isinstance(vector_store, PGVectorStore):
            vector_store.client.query(f"DELETE FROM vector_store WHERE id = '{ref_doc_id}'")
            storage_context.persist(persist_dir=cache_dir)
            return True
    return False


def reset_documents(index: VectorStoreIndex, storage_context: StorageContext) -> bool:
    logger.info("Resetting documents")
    vector_store = index._vector_store
    if isinstance(vector_store, PGVectorStore):
        try:
            vector_store.clear()
            logger.info("Vector store cleared successfully")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            persist_storage_context(storage_context)
            logger.info("Empty storage context persisted")
            return True
        except Exception as e:
            logger.error(f"Error resetting documents: {str(e)}", exc_info=True)
            return False
    else:
        logger.warning(f"Unsupported vector store type: {type(vector_store)}")
        return False
