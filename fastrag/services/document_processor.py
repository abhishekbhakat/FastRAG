import httpx
from bs4 import BeautifulSoup
from llama_index import Document

from fastrag.config import logger


def process_file(content: bytes, filename: str) -> Document:
    logger.info(f"Processing file: {filename}")
    try:
        text_content = content.decode("utf-8")
        logger.debug(f"Decoded {len(text_content)} characters from {filename}")
        document = Document(text=text_content, metadata={"source": filename})
        logger.info(f"Successfully processed file: {filename}")
        return document
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
        raise


async def process_url(url: str) -> Document:
    logger.info(f"Processing URL: {url}")
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Fetching content from URL: {url}")
            response = await client.get(url)
            logger.debug(f"Received response from {url}, status code: {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text()
            logger.debug(f"Extracted {len(text_content)} characters of text from {url}")
        document = Document(text=text_content, metadata={"source": url})
        logger.info(f"Successfully processed URL: {url}")
        return document
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
        raise
