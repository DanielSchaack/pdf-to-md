from typing import List, Dict, Optional, Any
from .llm_interfaces import LLMProvider
from core.file_service import encode_image_to_base64
import logging

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory class to create instances of LLM providers.
    """

    @staticmethod
    def create_provider(provider_name: str,
                        api_url: Optional[str] = None,
                        api_key: Optional[str] = None) -> LLMProvider:
        """
        Creates and returns an instance of the specified LLM provider.

        Args:
            provider_name: The name of the provider (e.g., "openrouter").
            api_url: The API URL for the provider.
            api_key: The API key.

        Returns:
            An instance of a class that implements LLMProvider.

        Raises:
            ValueError: If the provider_name is not supported.
        """
        if provider_name.lower() == "openrouter":
            logger.debug("Returning openrouter")
            if not api_url:
                api_url = "https://openrouter.ai/api/v1/chat/completions"
            return OpenRouterProvider(api_url=api_url, api_key=api_key)
        # elif provider_name.lower() == "ollama":
        #     logger.debug("Returning ollama", exc_info=True)
        #     return AnotherProvider(api_url=api_url, api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")


class OpenRouterProvider(LLMProvider):
    """
    Concrete implementation for the OpenRouter.ai LLM provider.
    """

    def _prepare_request_payload(self,
                                 model: str,
                                 user_message: str,
                                 system_prompt: Optional[str] = None,
                                 image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepares the request payload specific to OpenRouter.ai.
        """

        messages: List[Dict[str, Any]] = []
        # Add system prompt if provided
        if system_prompt:
            logger.debug(f"Adding system prompt: {system_prompt}")
            messages.append({"role": "system", "content": system_prompt})

        # Prepare user content (text and optional image)
        user_content_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
        if image_path:
            try:
                logger.debug(f"Attempting to encode image at path: {image_path}")
                base64_image = encode_image_to_base64(image_path)
                image_format = image_path.split('.')[-1].lower()
                if image_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
                    logger.warning(f"Image format '{image_format}' from path '{image_path}' might not be universally supported by the model. Defaulting to 'image/png' MIME type.")
                    mime_type = "image/png"
                else:
                    mime_type = f"image/{image_format}"
                data_url = f"data:{mime_type};base64,{base64_image}"
                logger.debug(f"Encoded image as: {data_url}")
                user_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            except FileNotFoundError:
                logger.error(f"File not found at path '{image_path}'")
                raise
            except Exception as e:
                logger.error(f"Error encoding image at path '{image_path}': {e}")
                raise RuntimeError(f"Failed to process image at {image_path}")

        messages.append({"role": "user", "content": user_content_parts})

        payload = {
            "model": model,
            "messages": messages
        }

        logger.debug(f"Prepared request payload: {payload}")
        return payload

    def extract_response_message(self, response_json: Dict[str, Any]) -> str:
        """
        Extracts the assistant's message from the OpenRouter.ai response.
        """
        try:
            content = response_json["choices"][0]["message"]["content"]
            return str(content)
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error extracting message from response: {e}. Full response: {response_json}", exc_info=True)
            return "Error: Could not parse LLM response."
