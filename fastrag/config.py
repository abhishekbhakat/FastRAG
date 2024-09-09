import logging
import os

from dotenv import load_dotenv

load_dotenv()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
APP_NAME = "SuperMemPy"

# Construct the database URL using environment variables
DB_URL = f"postgresql://{os.environ.get('PGUSER')}:{os.environ.get('PGPASSWORD')}@{os.environ.get('PGHOST')}:{os.environ.get('PGPORT')}/{os.environ.get('PGDATABASE')}"

config = {
    "database_url": DB_URL,
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
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    return logging.getLogger(APP_NAME)

logger = setup_logging()
