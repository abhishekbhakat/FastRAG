from fastapi import APIRouter

from fastrag.config import config
from fastrag.services.llm import get_llm
from fastrag.services.storage_context import get_index
from fastrag.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/query")
async def query(query: str):
    # Get vector store
    vs = get_vector_store(config)

    # Load storage context and index
    index, _ = get_index(config=config, vector_store=vs, cache_dir=config["cache_dir"])

    # Query vector store
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    # Use LLM to generate response
    llm = get_llm(config)
    enhanced_response = llm.complete(f"Based on the following information, please provide a concise and informative answer to the query '{query}': {response}")

    return {"query": query, "response": enhanced_response.text}
