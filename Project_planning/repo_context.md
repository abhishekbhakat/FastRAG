# FastRAG

## Repository Information
* Root folder: `/Volumes/Passport616/AI/FastRAG`
* Current branch: `main`

## Repository Structure
<pre><code>
FastRAG
├── fastrag
│   ├── docs
│   │   └── fastrag_description.md
│   ├── routes
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── query.py
│   ├── services
│   │   ├── bootstrap.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── storage_context.py
│   │   └── vector_store.py
│   ├── __init__.py
│   ├── app_config.py
│   ├── config.py
│   ├── database.py
│   ├── default_page.py
│   ├── main.py
│   └── models.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Not_working.md
├── pyproject.toml
├── README.md
└── requirements.txt
</code></pre>

## File Contents

### .gitignore
<pre><code>
.aider*
.env
.cache/
.venv/

*.pyc
*.log
*.ruff_cache/</code></pre>

### .pre-commit-config.yaml
<pre><code>
repos:
-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
    -   id: ruff
        args: [--fix]
    -   id: ruff-format

-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
</code></pre>

### LICENSE
<pre><code>
MIT License

Copyright (c) 2024 Abhishek Bhakat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</code></pre>

### Not_working.md
<pre><code>
- After adding documents, they are not showing up in storage context.
- Needed status page to check if AI and embedding both are available and online
</code></pre>

### README.md
<pre><code>
# FastRAG: FastAPI and FastUI Web Application for RAG

FastRAG is a powerful and user-friendly web application that combines FastAPI and FastUI to provide a seamless experience for Retrieval-Augmented Generation (RAG). It offers an easy way to import, process, and query various data sources using a vector store backed by PostgreSQL.

## Features

- **Easy Data Import**: Directly import data from GitHub repositories, URLs, and various file types through the web UI.
- **Vector Store Integration**: Utilizes PostgreSQL as a vector store for efficient similarity searches.
- **FastAPI Backend**: Robust and high-performance API built with FastAPI.
- **FastUI Frontend**: Intuitive and responsive user interface powered by FastUI.
- **Docker Support**: Simple deployment using Docker for consistency across environments.
- **RAG Capabilities**: Perform advanced queries on your imported data using state-of-the-art RAG techniques.

## Prerequisites

- Docker and Docker Compose
- Git (for cloning the repository)
- Python 3.8+ (for local development)

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fastrag.git
   cd fastrag
   ```

2. Create a `.env` file in the project root and add your configuration:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/fastrag
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

4. Access the application:
   Open your web browser and navigate to `http://localhost:3000`

## Usage

1. **Importing Data**:
   - Use the web UI to import data from GitHub repositories, URLs, or by uploading files directly.
   - Supported file types include PDF, TXT, MD, and more.

2. **Querying**:
   - Once data is imported, use the query interface to ask questions or retrieve information.
   - The RAG system will process your query and return relevant results from the imported data.

3. **Managing Data**:
   - View, update, or delete imported data sources through the web interface.

## Configuration

- Environment variables can be set in the `.env` file or passed to Docker.
- Key configurations:
  - `DATABASE_URL`: PostgreSQL connection string
  - `OPENAI_API_KEY`: Your OpenAI API key for RAG functionality

## Development

To set up a development environment:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```
   uvicorn fastrag.main:app --reload --host 0.0.0.0 --port 3000
   ```

3. For frontend development, refer to the FastUI documentation for setup instructions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- FastAPI
- FastUI
- PostgreSQL
- OpenAI (for RAG capabilities)

For more information or support, please open an issue on the GitHub repository.
</code></pre>

### fastrag/__init__.py
<pre><code>
# This file can be left empty
</code></pre>

### fastrag/app_config.py
<pre><code>
from fastapi import FastAPI

app = FastAPI()
</code></pre>

### fastrag/config.py
<pre><code>
import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
LOG_LEVEL = logging.DEBUG  # Changed to DEBUG for more verbose logging
APP_NAME = "fastrag"

UPLOAD_DIRECTORY = os.environ.get("UPLOAD_DIRECTORY", "uploads")

# Construct the database URL using environment variables
# DB_URL = f"postgresql://{os.environ.get('PGUSER')}:{os.environ.get('PGPASSWORD')}@{os.environ.get('PGHOST')}:{os.environ.get('PGPORT')}/{os.environ.get('PGDATABASE')}"

config = {
    "database_url": os.environ.get("DATABASE_URL"),
    "embed_dimension": int(os.environ.get("EMBED_DIMENSION", 1536)),
    "embed_batch_size": int(os.environ.get("EMBED_BATCH_SIZE", 100)),
    "embeddings_provider": os.environ.get("EMBEDDING_PROVIDER"),
    "embedding_model": os.environ.get("EMBEDDING_MODEL"),
    "embedding_api_base": os.environ.get("EMBEDDING_API_BASE_URL"),
    "embedding_api_key": os.environ.get("OPENAI_API_KEY"),
    "llm_provider": os.environ.get("LLM_PROVIDER"),
    "llm_model": os.environ.get("LLM_MODEL"),
    "llm_api_base": os.environ.get("LLM_API_BASE_URL"),
    "llm_api_key": os.environ.get("LLM_API_KEY"),
    "cache_dir": os.environ.get("CACHE_DIR", ".cache"),
}


# Setup logging
def setup_logging():
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # File handler
    file_handler = RotatingFileHandler(
        f"{APP_NAME}.log", maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()
logger.info(f"Logging initialized for {APP_NAME}")
</code></pre>

### fastrag/database.py
<pre><code>
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from fastrag.config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
</code></pre>

### fastrag/default_page.py
<pre><code>
from fastui import AnyComponent, components as c
from fastui.events import GoToEvent, PageEvent

from fastrag.models import UploadForm, URLForm


def get_default_page() -> list[AnyComponent]:
    return [
        c.PageTitle(text="FastRAG"),
        c.Navbar(
            title="",
            title_event=GoToEvent(url="/"),
            start_links=[
                c.Link(components=[c.Text(text="Home")], on_click=GoToEvent(url="/"), active="startswith:/", mode="navbar"),
                c.Link(components=[c.Text(text="Data")], on_click=GoToEvent(url="/data"), active="startswith:/data", mode="navbar"),
            ],
        ),
        c.Page(components=get_page_components()),
        c.Footer(extra_text="RAG Chatbot powered by FastUI", links=[]),
    ]


def get_page_components() -> list[AnyComponent]:
    return [
        c.Div(
            components=[
                c.Heading(text="FastRAG", level=1),
                c.Paragraph(text="Upload documents, paste URLs, and chat with your data using LLM and vector database."),
            ]
        ),
        c.Div(
            components=[
                c.Heading(text="Upload Documents", level=2),
                c.ModelForm(
                    model=UploadForm,
                    submit_url="/api/upload_documents",
                    loading=[c.Spinner(text="Uploading ...")],
                    submit_trigger=PageEvent(name="upload_documents"),
                ),
            ],
            class_name="border-top mt-3 pt-1",
        ),
        c.Div(
            components=[
                c.Heading(text="Add URLs", level=2),
                c.ModelForm(model=URLForm, submit_url="/api/add_url", submit_trigger=PageEvent(name="add_url")),
            ],
            class_name="border-top mt-3 pt-1",
        ),
        c.Toast(title="Document Uploaded", body=[c.Paragraph(text="Successfully processed the document.")], open_trigger=PageEvent(name="document-upload-success"), position="bottom-center"),
        c.Toast(title="Document Upload Failed", body=[c.Paragraph(text="Failed to process document.")], open_trigger=PageEvent(name="docuent-upload-failed"), position="bottom-center"),
        c.Toast(title="Add URL Success", body=[c.Paragraph(text="Successfully added URL.")], open_trigger=PageEvent(name="add_url_success"), position="bottom-center"),
        c.Toast(title="Add URL Failed", body=[c.Paragraph(text="Failed to add URL.")], open_trigger=PageEvent(name="add_url_failed"), position="bottom-center"),
    ]
</code></pre>

### fastrag/docs/fastrag_description.md
<pre><code>
# FastRAG: Rapid Retrieval-Augmented Generation Application

FastRAG is a powerful and efficient Retrieval-Augmented Generation (RAG) application built with FastAPI. It combines the strengths of large language models (LLMs) with a robust document retrieval system to provide accurate and context-aware responses to user queries.

## Key Components and Responsibilities

1. **Document Ingestion**
   - Processes various document formats (PDF, DOCX, plain text)
   - Extracts text content from documents
   - Creates embeddings for efficient retrieval
   - Stores document information in a vector database

2. **Vector Store**
   - Utilizes PostgreSQL with pgvector for efficient similarity search
   - Manages storage and retrieval of document embeddings

3. **Embedding Generation**
   - Supports multiple embedding providers (OpenAI, Ollama, LMStudio)
   - Generates high-quality embeddings for both documents and queries

4. **Query Processing**
   - Receives user queries through a chat interface
   - Converts queries into embeddings for similarity search
   - Retrieves relevant documents based on query embeddings

5. **Language Model Integration**
   - Integrates with various LLM providers (OpenAI, Anthropic, etc.)
   - Generates context-aware responses using retrieved documents and user queries

6. **API and User Interface**
   - Provides a FastAPI-based backend for efficient request handling
   - Offers a user-friendly interface for document upload, URL ingestion, and chatting

7. **Caching and Performance Optimization**
   - Implements caching mechanisms for faster response times
   - Optimizes embedding and retrieval processes for scalability

8. **Logging and Monitoring**
   - Maintains comprehensive logs for debugging and performance analysis
   - Monitors system health and performance metrics

FastRAG aims to provide a seamless experience for users to interact with their document collection, leveraging the power of modern NLP techniques and efficient information retrieval systems.
</code></pre>

### fastrag/main.py
<pre><code>
import asyncio
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse
from fastui import AnyComponent, FastUI, components as c, prebuilt_html
from fastui.events import GoToEvent, PageEvent
from fastui.forms import fastui_form

from fastrag.app_config import app
from fastrag.config import logger
from fastrag.default_page import get_default_page
from fastrag.models import ChatForm, MessageHistoryModel, UploadForm, URLForm
from fastrag.routes import ingest, query
from fastrag.services.bootstrap import bootstrap_app

app.message_history = []

logger.info("Including routers")
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application")
    try:
        bootstrap_app()
        logger.info("Application bootstrapped successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        # You might want to raise the exception here if you want to prevent the app from starting
        # raise e


@app.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def api_index():
    logger.debug("Handling API index request")
    return get_default_page()


@app.post("/api/upload_documents", response_model=FastUI, response_model_exclude_none=True)
async def upload_documents(file: UploadFile = File(...)) -> list[AnyComponent]:
    result = await ingest.ingest_document(file)
    result = {"status": "success"}
    if result["status"] == "success":
        return [
            c.FireEvent(event=PageEvent(name="document-upload-success")),
            c.ModelForm(
                model=UploadForm,
                submit_url="/api/upload_documents",
                loading=[c.Spinner(text="Uploading ...")],
                submit_trigger=PageEvent(name="upload_documents"),
            ),
        ]
    else:
        return [
            c.FireEvent(event=PageEvent(name="document-upload-failed")),
            c.ModelForm(
                model=UploadForm,
                submit_url="/api/upload_documents",
                loading=[c.Spinner(text="Uploading ...")],
                submit_trigger=PageEvent(name="upload_documents"),
            ),
        ]


@app.post("/api/add_url", response_model=FastUI, response_model_exclude_none=True)
async def add_url(form: Annotated[URLForm, fastui_form(URLForm)]):
    logger.debug("Ingest URL: " + form.url)
    result = await ingest.ingest_url(form.url)
    result = {"status": "success"}
    if result["status"] == "success":
        return [
            c.FireEvent(event=PageEvent(name="add_url_success")),
            c.ModelForm(
                model=URLForm,
                submit_url="/api/add_url",
                submit_trigger=PageEvent(name="add_url"),
            ),
        ]
    else:
        return [
            c.FireEvent(event=PageEvent(name="add_url_failed")),
            c.ModelForm(
                model=URLForm,
                submit_url="/api/add_url",
                submit_trigger=PageEvent(name="add_url"),
            ),
        ]


@app.post("/api/chat", response_model=FastUI, response_model_exclude_none=True)
async def chat(chat_form: Annotated[ChatForm, fastui_form(ChatForm)]):
    response = await query.query(chat_form.message)
    app.message_history.append(MessageHistoryModel(message=f"User: {chat_form.message}"))
    app.message_history.append(MessageHistoryModel(message=f"Chatbot: {response['response']}"))
    return [c.Markdown(text=response["response"]), c.FireEvent(event=GoToEvent(url="/"))]


async def chat_response_generator(message: str) -> AsyncIterable[str]:
    response = await query.query(message)
    app.message_history.append(MessageHistoryModel(message=f"User: {message}"))
    app.message_history.append(MessageHistoryModel(message=f"Chatbot: {response['response']}"))
    m = FastUI(root=[c.Markdown(text=response["response"])])
    msg = f"data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
    yield msg
    while True:
        yield msg
        await asyncio.sleep(10)


@app.get("/{path:path}")
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title="FastRAG"))
</code></pre>

### fastrag/models.py
<pre><code>
from typing import Annotated

from fastapi import UploadFile
from fastui.forms import FormFile
from pydantic import BaseModel, Field


class UploadForm(BaseModel):
    file: Annotated[UploadFile, FormFile(accept="*/*")] = Field(title="File")


class ChatForm(BaseModel):
    message: str = Field(title="Message")


class URLForm(BaseModel):
    url: str = Field(title="URL", min_length=1)


class MessageHistoryModel(BaseModel):
    message: str
</code></pre>

### fastrag/routes/__init__.py
<pre><code>
# This file can be left empty
</code></pre>

### fastrag/routes/ingest.py
<pre><code>
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
</code></pre>

### fastrag/routes/query.py
<pre><code>
from fastapi import APIRouter

from fastrag.config import config, logger
from fastrag.services.llm import get_llm
from fastrag.services.storage_context import get_index
from fastrag.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/query")
async def query(query: str):
    logger.info(f"Received query: {query}")
    try:
        # Get vector store
        logger.debug("Getting vector store")
        vs = get_vector_store(config)

        # Load storage context and index
        logger.debug("Loading storage context and index")
        index, _ = get_index(config=config, vector_store=vs, cache_dir=config["cache_dir"])

        # Query vector store
        logger.debug("Querying vector store")
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        logger.debug(f"Vector store response: {response}")

        # Use LLM to generate response
        logger.debug("Getting LLM")
        llm = get_llm(config)
        logger.debug("Generating enhanced response with LLM")
        context = f"You are an AI assistant for the FastRAG application. Use the following information to answer the user's query: {response}"
        enhanced_response = llm.complete(f"{context}\n\nUser query: {query}\n\nAI assistant:")
        logger.info(f"Generated response for query: {query}")

        return {"query": query, "response": enhanced_response.text}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {"query": query, "response": f"Error processing query: {str(e)}"}
</code></pre>

### fastrag/services/bootstrap.py
<pre><code>
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
</code></pre>

### fastrag/services/embeddings.py
<pre><code>
from typing import Any

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from fastrag.config import logger


def get_openai_embedding(config: dict[str, Any]) -> OpenAIEmbedding:
    logger.info("Initializing OpenAI embedding")
    return OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])


def get_ollama_embedding(config: dict[str, Any]) -> OllamaEmbedding:
    logger.info("Initializing Ollama embedding")
    return OllamaEmbedding(model_name=config["embedding_model"])


def get_lmstudio_embedding(config: dict[str, Any]) -> OpenAIEmbedding:
    logger.info("Initializing LMStudio embedding")
    return OpenAIEmbedding(api_key=config["embedding_api_key"], api_base=config.get("embedding_api_base"), model_name=config["embedding_model"])


def get_embedding_model(config: dict[str, Any]) -> OpenAIEmbedding | OllamaEmbedding:
    logger.info("Getting embedding model")
    embedding_provider = config["embeddings_provider"]
    logger.debug(f"Embedding provider: {embedding_provider}")
    try:
        if embedding_provider == "openai":
            return get_openai_embedding(config)
        elif embedding_provider == "ollama":
            return get_ollama_embedding(config)
        elif embedding_provider == "lmstudio":
            return get_lmstudio_embedding(config)
        else:
            logger.error(f"Unsupported embeddings provider: {embedding_provider}")
            raise ValueError(f"Unsupported embeddings provider: {embedding_provider}")
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        raise
</code></pre>

### fastrag/services/llm.py
<pre><code>
from typing import Any

from llama_index.llms.litellm import LiteLLM

from fastrag.config import config, logger


async def generate_response(query: str, relevant_docs: list) -> str:
    logger.info(f"Generating response for query: {query}")
    try:
        model = config["llm_model"]
        provider = config["llm_provider"]
        logger.debug(f"Using LLM model: {model}, provider: {provider}")
        llm = LiteLLM(model=model, provider=provider)
        context = "\n".join(relevant_docs)
        prompt = f"Based on the following context, answer the question: {query}\n\nContext: {context}"
        logger.debug(f"Sending prompt to LLM: {prompt[:100]}...")  # Log first 100 chars of prompt
        response = await llm.acomplete(prompt)
        logger.info("Response generated successfully")
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise


def get_llm(config: dict[str, Any]) -> LiteLLM:
    logger.info("Initializing LLM")
    try:
        llm_provider = config["llm_provider"]
        model = config.get("llm_model", "gpt-3.5-turbo")
        logger.debug(f"LLM provider: {llm_provider}, model: {model}")

        api_key = config.get("llm_api_key")
        api_base = config.get("llm_api_base")

        litellm_kwargs = {
            "model": model,
            "api_key": api_key,
        }

        if llm_provider not in ["openai", "deepseek", "anthropic", "huggingface"]:
            litellm_kwargs["api_base"] = api_base

        # Remove None values
        litellm_kwargs = {k: v for k, v in litellm_kwargs.items() if v is not None}
        logger.debug(f"LiteLLM kwargs: {litellm_kwargs}")
        llm = LiteLLM(**litellm_kwargs)
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
        raise
</code></pre>

### fastrag/services/storage_context.py
<pre><code>
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
        storage_context.persist(persist_dir=cache_dir)


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
    return index


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
</code></pre>

### fastrag/services/vector_store.py
<pre><code>
from typing import Any
from urllib.parse import urlparse

from llama_index.vector_stores.postgres import PGVectorStore

from fastrag.config import logger


def get_vector_store(config: dict[str, Any]) -> PGVectorStore:
    logger.info("Initializing vector store")
    try:
        # Parse the database URL
        url = urlparse(config["database_url"])
        logger.debug(f"Parsed database URL: {url.scheme}://{url.hostname}:{url.port}{url.path}")

        # Extract connection details
        username = url.username
        password = url.password
        host = url.hostname
        port = url.port
        database = url.path[1:]  # Remove the leading '/'

        # Create and return the PGVectorStore
        vector_store = PGVectorStore.from_params(
            database=database,
            host=host,
            password=password,
            port=port,
            user=username,
            table_name="vector_store",
            embed_dim=config["embed_dimension"],
        )
        logger.info("Vector store initialized successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
        raise
</code></pre>

### pyproject.toml
<pre><code>
[project]
name = "fastrag"
version = "0.1.0"
description = "FastAPI and FastUI Web Application for RAG"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
dependencies = [
    "fastapi",
    "fastui[fastapi]",
    "uvicorn",
    "psycopg2-binary",
    "llama-index",
    "llama-index-readers-file",
    "llama-index-vector-stores-postgres",
    "llama-index-embeddings-openai",
    "llama-index-embeddings-ollama",
    "llama-index-embeddings-bedrock",
    "llama-index-llms-litellm",
    "pydantic",
    "PyYAML",
    "rich",
    "python-dotenv",
    "python-multipart",
    "pypdf",
    "python-docx",
]
requires-python = ">=3.8"
readme = "README.md"
license = {file = "LICENSE"}

[project.urls]
Homepage = "https://github.com/yourusername/fastrag"
Repository = "https://github.com/yourusername/fastrag.git"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["fastrag"]

[tool.uv.pip]
# Note: Use -U flag for eager upgrades, e.g., `uv pip install -U package_name`


[tool.ruff]
line-length = 200
indent-width = 4
fix = true
preview = true

lint.select = [
    "E",  # pycodestyle errors
    "F",  # pyflakes
    "I",  # isort
    "W",  # pycodestyle warnings
    "C90",  # Complexity
    "C",  # flake8-comprehensions
    "ISC",  # flake8-implicit-str-concat
    "T10",  # flake8-debugger
    "A",  # flake8-builtins
    "UP",  # pyupgrade
]

lint.ignore = [
    "C416",
    "C408"
]

lint.fixable = ["ALL"]
lint.unfixable = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false

[tool.ruff.lint.isort]
combine-as-imports = true
</code></pre>

### requirements.txt
<pre><code>
# FastAPI and FastUI
fastapi
fastui[fastapi]
uvicorn

# Database
psycopg2-binary

# LlamaIndex and related packages
llama-index
llama-index-readers-file
llama-index-vector-stores-postgres
llama-index-embeddings-openai
llama-index-embeddings-ollama
llama-index-embeddings-bedrock
llama-index-llms-litellm

# Utilities
pydantic
PyYAML
rich
python-dotenv
python-multipart

# Document processing
pypdf
python-docx
</code></pre>

