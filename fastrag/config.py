import logging
import os
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv

load_dotenv()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
LOG_LEVEL = logging.DEBUG  # Changed to DEBUG for more verbose logging
APP_NAME = "SuperMemPy"

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
