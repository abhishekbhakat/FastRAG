from fastapi import APIRouter
from fastrag.models import ServiceStatus

router = APIRouter()

@router.get("/health")
async def get_health_status():
    llm_status = ServiceStatus.get(service_name="LLM")
    embedding_status = ServiceStatus.get(service_name="Embedding")
    vector_store_status = ServiceStatus.get(service_name="VectorStore")

    return {
        "LLM": llm_status.status,
        "Embedding": embedding_status.status,
        "VectorStore": vector_store_status.status,
    }
