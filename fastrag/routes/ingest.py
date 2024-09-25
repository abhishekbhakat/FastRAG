import traceback
from pathlib import Path

import httpx
from fastapi import APIRouter, File, UploadFile
from fastui import FastUI, components as c
from fastui.components.display import DisplayLookup
from fastui.events import PageEvent
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.readers.file import DocxReader, PDFReader
from llama_index.vector_stores.postgres import PGVectorStore
from pydantic import BaseModel, Field

from fastrag.config import config, logger
from fastrag.services.storage_context import create_index, get_all_documents, persist_storage_context

router = APIRouter()


class Loaded_Document(BaseModel):
    doc_type: str = Field(title="Type")
    name: str = Field(title="Name")
    doc_id: str = Field(title="Document ID")


@router.post("/ingest/document", response_model=FastUI, response_model_exclude_none=True)
async def ingest_document(file: UploadFile = File(...)):
    logger.info(f"Starting ingestion for document: {file.filename}")
    original_filename = file.filename
    try:
        # Read file content
        content = await file.read()
        logger.debug(f"Read {len(content)} bytes from {file.filename}")

        local_upload_dir = config.get("upload_dir")

        # Save the file to the local directory
        file_path = Path(local_upload_dir) / original_filename
        file_path.write_bytes(content)

        logger.debug(f"Saved content to local file: {file_path}")

        # Determine file type and process accordingly
        if file.filename.lower().endswith(".pdf"):
            logger.info("Processing PDF file")
            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=str(file_path))
        elif file.filename.lower().endswith(".docx"):
            logger.info("Processing DOCX file")
            docx_reader = DocxReader()
            documents = docx_reader.load_data(file=str(file_path))
        else:
            logger.info("Processing as plain text")
            documents = [Document(text=content.decode("utf-8", errors="ignore"), extra_info={"file_path": file.filename})]

        logger.info(f"Processed {len(documents)} documents from {file.filename}")

        index: VectorStoreIndex = config.get("index")
        vs: PGVectorStore = config["vector_store"]
        sc: StorageContext = config.get("storage_context")

        if index is None:
            logger.info("Index not found, creating new index from documents")
            index, sc = create_index(documents=documents, vector_store=vs)
            config["index"] = index
            config["storage_context"] = sc
        else:
            for i, document in enumerate(documents, start=1):
                logger.debug(f"Inserting document {i}/{len(documents)} into index")
                index.insert(document)

        logger.info("Persisting storage context")
        persist_storage_context(sc)

        logger.info(f"Successfully processed and ingested {file.filename}")
        return [c.FireEvent(event=PageEvent(name="document-upload-success"))]
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}", exc_info=True)
        return [c.FireEvent(event=PageEvent(name="document-upload-failed"))]


@router.post("/ingest/url", response_model=FastUI, response_model_exclude_none=True)
async def ingest_url(url: str):
    logger.info(f"Starting ingestion for URL: {url}")

    try:
        # Fetch content from URL using httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://r.jina.ai/{url}")
            response.raise_for_status()
            content = response.text

        # Create a Document object
        document = Document(text=content, extra_info={"url": url})

        index: VectorStoreIndex = config.get("index")
        vs: PGVectorStore = config["vector_store"]
        sc: StorageContext = config.get("storage_context")

        if index is None:
            logger.info("Index not found, creating new index from document")
            index, sc = create_index(documents=[document], vector_store=vs)
            config["index"] = index
            config["storage_context"] = sc
        else:
            logger.debug("Inserting document into index")
            index.insert(document)

        persist_storage_context(sc)

        logger.info(f"Successfully processed and ingested URL: {url}")
        return [c.FireEvent(event=PageEvent(name="add_url_success"))]

    except Exception as e:
        logger.error(f"Error during URL ingestion: {str(e)}", exc_info=True)
        return [c.FireEvent(event=PageEvent(name="add_url_failed"))]


@router.get("/get_documents", response_model=FastUI, response_model_exclude_none=True)
async def get_documents():
    logger.info("Fetching all documents")
    try:
        index: VectorStoreIndex = config.get("index")
        logger.debug(f"Index: {index}")
        if index is None:
            logger.warning("Index not found, returning empty table")
            return [c.Paragraph(text="No documents found. The index is empty.")]

        documents = get_all_documents(index)
        logger.info(f"Retrieved {len(documents)} documents")
        # logger.debug(f"Documents: {documents}")

        if not documents:
            return [c.Paragraph(text="No documents found in the index.")]

        # Find unique filename or URLs

        table_data = [Loaded_Document(doc_type="URL" if doc.get("url") else "File", name=doc.get("filename", doc.get("url", "N/A")), doc_id=doc.get("ref_doc_id", "N/A")) for doc in documents]
        # logger.debug(f"Table data: {table_data}")

        result = [
            c.Table(
                data=table_data,
                data_model=Loaded_Document,
                columns=[
                    DisplayLookup(field="doc_type", title="Type"),
                    DisplayLookup(field="name", title="Name"),
                    DisplayLookup(field="doc_id", title="Document ID"),
                ],
            )
        ]
        # logger.debug(f"Returning result: {result}")
        return result
    except Exception as e:
        error_message = f"Error fetching documents: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_message)
        return [c.Paragraph(text=f"Error fetching documents: {str(e)}")]


@router.post("/reset_documents", response_model=FastUI, response_model_exclude_none=True)
async def reset_documents():
    logger.info("Resetting all documents")
    try:
        index: VectorStoreIndex = config.get("index")
        vs: PGVectorStore = config["vector_store"]
        sc: StorageContext = config.get("storage_context")

        if index is not None:
            # Clear the index
            index.delete_ref_doc("*", delete_from_docstore=True)

            # Clear the vector store
            vs.clear()

            # Create a new empty index
            index, sc = create_index(documents=[], vector_store=vs)

            # Update the config
            config["index"] = index
            config["storage_context"] = sc

            logger.info("Successfully reset all documents")
            return [c.FireEvent(event=c.PageEvent(name="documents-reset"))]
        else:
            logger.warning("No index found, nothing to reset")
            return [c.Paragraph(text="No documents to reset")]
    except Exception as e:
        error_message = f"Error resetting documents: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_message)
        return [c.Paragraph(text=f"Error resetting documents: {str(e)}")]
