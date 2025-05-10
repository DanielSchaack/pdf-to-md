from typing import List, Dict, Optional, Any
from .llm_interfaces import LLMProvider
from .llm_utils import encode_image_to_base64
import logging

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """
    Concrete implementation for the OpenRouter.ai LLM provider.
    """

    def _prepare_request_payload(self, user_message: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepares the request payload specific to OpenRouter.ai.
        """
        messages: List[Dict[str, Any]] = []

        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Prepare user content (text and optional image)
        user_content_parts: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]

        if image_path:
            try:
                base64_image = encode_image_to_base64(image_path)
                image_format = image_path.split('.')[-1].lower()
                if image_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
                    logger.warning(f"Image format '{image_format}' from path '{image_path}' might not be universally supported by the model. Defaulting to 'image/jpeg' MIME type.")
                    mime_type = "image/png"
                else:
                    mime_type = f"image/{image_format}"

                data_url = f"data:{mime_type};base64,{base64_image}"
                user_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            except FileNotFoundError:
                raise
            except Exception as e:
                logger.error(f"Error encoding image at path '{image_path}': {e}", exc_info=True)
                raise RuntimeError(f"Failed to process image at {image_path}")

        messages.append({"role": "user", "content": user_content_parts})

        payload = {
            "model": self.model,
            "messages": messages
        }
        return payload

    def extract_response_message(self, response_json: Dict[str, Any]) -> str:
        """
        Extracts the assistant's message from the OpenRouter.ai response.
        """
        try:
            # Based on the expected response format:
            # {
            #   "choices": [ { "message": { "content": "..." } } ]
            #   ...
            # }
            content = response_json["choices"][0]["message"]["content"]
            return str(content)
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error extracting message from response: {e}. Full response: {response_json}", exc_info=True)
            return "Error: Could not parse LLM response."
