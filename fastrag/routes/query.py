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
