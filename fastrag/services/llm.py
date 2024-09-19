from typing import Any

from llama_index.llms.litellm import LiteLLM

from fastrag.config import config, logger
from fastrag.models import ServiceStatus


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
        save_llm_status("online")
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        save_llm_status("offline")
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
        save_llm_status("online")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
        save_llm_status("offline")
        raise


def save_llm_status(status: str):
    from datetime import datetime
    ServiceStatus(service_name="LLM", status=status, last_checked=datetime.now().isoformat()).save()
