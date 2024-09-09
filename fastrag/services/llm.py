from typing import Any
from llama_index.llms.litellm import LiteLLM
from fastrag.config import config

async def generate_response(query: str, relevant_docs: list) -> str:
    model = config["llm_model"]
    provider = config["llm_provider"]
    llm = LiteLLM(model=model, provider=provider)
    context = "\n".join(relevant_docs)
    prompt = f"Based on the following context, answer the question: {query}\n\nContext: {context}"
    response = await llm.acomplete(prompt)
    return response.text

def get_llm(config: dict[str, Any]) -> LiteLLM:
    llm_provider = config["llm_provider"]
    model = config.get("llm_model", "gpt-3.5-turbo")

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
    return LiteLLM(**litellm_kwargs)
