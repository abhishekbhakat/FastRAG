import tempfile
from pathlib import Path

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from llama_index.core import Document
from llama_index.readers.file import DocxReader, PDFReader

from fastrag.config import config, logger
from fastrag.services.storage_context import create_index, get_index
from fastrag.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/ingest/document")
async def ingest_document(file: UploadFile = File(...)):
    logger.info(f"Starting ingestion for document: {file.filename}")
    try:
        # Read file content
        content = await file.read()
        logger.debug(f"Read {len(content)} bytes from {file.filename}")

        # Save the uploaded file content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = Path(temp_file.name)
        logger.debug(f"Saved content to temporary file: {temp_file_path}")

        # Get vector store
        vs = get_vector_store(config)
        logger.debug("Retrieved vector store")

        # Load storage context and index
        index, sc = get_index(config=config, vector_store=vs, cache_dir=config["cache_dir"])
        logger.debug("Loaded storage context and index")

        # Determine file type and process accordingly
        if file.filename.lower().endswith(".pdf"):
            logger.info("Processing PDF file")
            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=temp_file_path)
        elif file.filename.lower().endswith(".docx"):
            logger.info("Processing DOCX file")
            docx_reader = DocxReader()
            documents = docx_reader.load_data(file=temp_file_path)
        else:
            logger.info("Processing as plain text")
            documents = [Document(text=content.decode("utf-8", errors="ignore"), extra_info={"file_path": file.filename})]

        logger.info(f"Processed {len(documents)} documents from {file.filename}")

        if not index:
            logger.info("Index not found, creating new index from documents")
            index = create_index(config=config, documents=documents, vector_store=vs, cache_dir=config["cache_dir"])
        else:
            for i, document in enumerate(documents):
                logger.debug(f"Inserting document {i + 1}/{len(documents)} into index")
                index.insert(document)

        logger.info("Persisting storage context")
        sc.persist(persist_dir=config["cache_dir"])

        logger.info(f"Successfully processed and ingested {file.filename}")
        return {
            "message": f"Successfully processed and ingested {file.filename}",
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}", exc_info=True)
        return {
            "message": f"Error processing {file.filename}: {str(e)}",
            "status": "error",
        }


@router.post("/ingest/url")
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

        # Get vector store
        vs = get_vector_store(config)
        logger.debug("Retrieved vector store")

        # Load storage context and index
        index, sc = get_index(config=config, vector_store=vs, cache_dir=config["cache_dir"])
        logger.debug("Loaded storage context and index")

        if not index:
            logger.info("Index not found, creating new index from document")
            index = create_index(config=config, documents=[document], vector_store=vs, cache_dir=config["cache_dir"])
        else:
            logger.debug("Inserting document into index")
            index.insert(document)

        logger.info("Persisting storage context")
        sc.persist(persist_dir=config["cache_dir"])

        logger.info(f"Successfully processed and ingested URL: {url}")
        return {
            "message": f"Successfully processed and ingested URL: {url}",
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error during URL ingestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing URL {url}: {str(e)}")
