from llama_index.embeddings.openai import OpenAIEmbedding
from fastrag.config import settings

def get_embed_model():
    return OpenAIEmbedding(model=settings.EMBED_MODEL)
