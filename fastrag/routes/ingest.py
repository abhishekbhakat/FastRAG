import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from llama_index.core import Document
from llama_index.readers.file import DocxReader, PDFReader

from fastrag.config import config
from fastrag.services.storage_context import get_index
from fastrag.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/ingest/document")
async def ingest_document(file: UploadFile = File(...)):
    # Read file content
    content = await file.read()

    # Save the uploaded file content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(content)
        temp_file_path = Path(temp_file.name)

    # Get vector store
    vs = get_vector_store(config)

    # Load storage context and index
    index, sc = get_index(config=config, vector_store=vs, cache_dir=config["cache_dir"])

    # Determine file type and process accordingly
    if file.filename.lower().endswith(".pdf"):
        pdf_reader = PDFReader()
        documents = pdf_reader.load_data(file=temp_file_path)
    elif file.filename.lower().endswith(".docx"):
        docx_reader = DocxReader()
        documents = docx_reader.load_data(file=temp_file_path)
    else:
        # For other file types, treat as plain text
        documents = [Document(text=content.decode("utf-8", errors="ignore"), extra_info={"file_path": file.filename})]

    for document in documents:
        index.insert(document)

    sc.persist(persist_dir=config["cache_dir"])

    return {
        "message": f"Successfully processed and ingested {file.filename}",
        "status": "success",
    }


@router.post("/ingest/url")
async def ingest_url(url: str):
    # Fetch content from URL
    # Create embeddings
    # Save to vector store
    # Return success response
    pass
