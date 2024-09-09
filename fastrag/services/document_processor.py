import httpx
from bs4 import BeautifulSoup
from llama_index import Document


def process_file(content: bytes, filename: str) -> Document:
    text_content = content.decode("utf-8")
    return Document(text=text_content, metadata={"source": filename})


async def process_url(url: str) -> Document:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text()
    return Document(text=text_content, metadata={"source": url})
