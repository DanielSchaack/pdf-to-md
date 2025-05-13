from core.llm_providers import LLMProviderFactory, LLMProvider
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)


class LlmService:
    def __init__(self,
                 provider: str = None,
                 image_model: str = None,
                 text_model: str = None,
                 api_url: str = None,
                 api_key: str = None):
        if not provider:
            provider = os.getenv("API_PROVIDER")
            assert provider, "A provider is required to be configured"
        if not image_model:
            image_model = os.getenv("API_IMAGE_MODEL")
        if not text_model:
            text_model = os.getenv("API_TEXT_MODEL")
        if not api_url:
            api_url = os.getenv("API_URL")
        if not api_key:
            api_key = os.getenv("API_KEY")
        self.llm_provider: LLMProvider = LLMProviderFactory.create_provider(
            provider_name=provider,
            image_model=image_model,
            text_model=text_model,
            api_url=api_url,
            api_key=api_key
        )

    def call_image_llm_provider(self,
                                image_path: str,
                                system_prompt: str,
                                user_message: str) -> str:
        try:
            logger.info(f"User: {user_message}")
            response_message = self.llm_provider.get_completion(model=self.llm_provider.image_model,
                                                                user_message=user_message,
                                                                system_prompt=system_prompt,
                                                                image_path=image_path)
            logger.info(f"LLM: {response_message}")

            return response_message

        except Exception as e:
            logger.error(e, exc_info=True)
            raise

    def call_text_llm_provider(self,
                               user_message: str,
                               system_prompt: Optional[str]) -> str:
        try:
            logger.debug(f"User: {user_message}")
            response_message = self.llm_provider.get_completion(model=self.llm_provider.text_model,
                                                                user_message=user_message,
                                                                system_prompt=system_prompt,
                                                                image_path=None)
            logger.info(f"LLM: {response_message}")
            return response_message

        except Exception as e:
            logger.error(e, exc_info=True)
            raise

