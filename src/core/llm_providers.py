from typing import List, Dict, Optional, Any
from core.file_service import encode_image_to_base64
from abc import ABC, abstractmethod
import requests
import json
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Defines the contract for sending requests and extracting responses.
    """

    def __init__(self,
                 image_model: str,
                 text_model: str,
                 api_url: str,
                 api_key: str):
        """
        Initializes the LLM provider.

        Args:
            api_url: The API endpoint URL for the provider.
            api_key: The API key for authentication.
        """
        self.image_model = image_model
        self.text_model = text_model
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    def _prepare_request_payload(self,
                                 model: str,
                                 user_message: str,
                                 system_prompt: Optional[str],
                                 image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepares the provider-specific request payload.
        This method must be implemented by concrete provider classes.

        Args:
            model: The name of the LLM model to use.
            user_message: The user's message or prompt.
            system_prompt: An optional system message to guide the LLM's behavior.
            image_path: Optional path to an image file for vision models.

        Returns:
            A dictionary representing the request payload.
        """
        pass

    @abstractmethod
    def extract_response_message(self, response_json: Dict[str, Any]) -> str:
        """
        Extracts the meaningful content from the provider's JSON response.
        This method must be implemented by concrete provider classes.

        Args:
            response_json: The JSON response from the LLM provider.

        Returns:
            The extracted text message from the LLM's response.
        """
        pass

    def send_request(self,
                     model: str,
                     user_message: str,
                     system_prompt: Optional[str] = None,
                     image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a request to the LLM provider.

        Args:
            model: The name of the LLM model to use.
            user_message: The user's message or prompt.
            system_prompt: An optional system message to guide the LLM's behavior.
            image_path: Optional path to an image file for vision models.

        Returns:
            The JSON response from the LLM provider as a dictionary.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
            ValueError: If the response is not valid JSON.
        """
        payload = self._prepare_request_payload(model=model,
                                                user_message=user_message,
                                                system_prompt=system_prompt,
                                                image_path=image_path)

        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err} - {response.text}", exc_info=True)
            raise
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error occurred: {req_err}", exc_info=True)
            raise
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response: {json_err}. Response text: {response.text}", exc_info=True)
            raise ValueError("Invalid JSON response from server.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during send_request: {e}", exc_info=True)
            raise ValueError(f"An unexpected error occurred: {e}")

    def get_completion(self,
                       model: str,
                       user_message: str,
                       system_prompt: Optional[str] = None,
                       image_path: Optional[str] = None) -> str:
        """
        A convenience method that sends a request and extracts the response message.

        Args:
            user_message: The user's message or prompt.
            image_path: Optional path to an image file for vision models.

        Returns:
            The extracted text message from the LLM's response.
        """
        response_json = self.send_request(model=model,
                                          user_message=user_message,
                                          system_prompt=system_prompt,
                                          image_path=image_path)
        return self.extract_response_message(response_json)


class LLMProviderFactory:
    """
    Factory class to create instances of LLM providers.
    """

    @staticmethod
    def create_provider(provider_name: str,
                        image_model: str = None,
                        text_model: str = None,
                        api_url: Optional[str] = None,
                        api_key: Optional[str] = None) -> LLMProvider:
        """
        Creates and returns an instance of the specified LLM provider.

        Args:
            provider_name: The name of the provider (e.g., "openrouter").
            image_model.
            text_model.
            api_url: The API URL for the provider.
            api_key: The API key.

        Returns:
            An instance of a class that implements LLMProvider.

        Raises:
            ValueError: If the provider_name is not supported.
        """
        if provider_name.lower() == "openrouter":
            assert api_key, "An api key is required to make use of openrouter"
            if not image_model:
                image_model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
                logger.warn(f"Manually set openrouter text model to {image_model}")
            if not text_model:
                text_model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
                logger.warn(f"Manually set openrouter text model to {text_model}")
            if not api_url:
                api_url = "https://openrouter.ai/api/v1/chat/completions"
                logger.warn(f"Manually set openrouter URL to {api_url}")
            logger.debug("Returning openrouter")
            return OpenRouterProvider(image_model=image_model, text_model=text_model, api_url=api_url, api_key=api_key)
        elif provider_name.lower() == "ollama":
            logger.debug("Returning ollama")
            # assert api_key, "An api key is required to make use of openrouter"
            if not image_model:
                image_model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
                logger.warn(f"Manually set ollama text model to {image_model}")
            if not text_model:
                text_model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
                logger.warn(f"Manually set ollama text model to {text_model}")
            if not api_url:
                api_url = "https://openrouter.ai/api/v1/chat/completions"
                logger.warn(f"Manually set openrouter URL to {api_url}")
            logger.debug("Returning openrouter")
            return OllamaProvider(image_model=image_model, text_model=text_model, api_url=api_url, api_key=api_key)
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
                user_content_parts.append({"type": "image_url",
                                           "image_url": {"url": data_url}
                                           })
            except FileNotFoundError:
                logger.error(f"File not found at path '{image_path}'")
                raise
            except Exception as e:
                logger.error(f"Error encoding image at path '{image_path}': {e}")
                raise RuntimeError(f"Failed to process image at {image_path}")

        messages.append({"role": "user", "content": user_content_parts})

        payload = {"model": model,
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


class OllamaProvider(LLMProvider):
    """
    Concrete implementation for the Ollama LLM provider,
    targeting an API endpoint similar to /api/generate.
    """

    def _prepare_request_payload(self,
                                 model: str,
                                 user_message: str,
                                 system_prompt: Optional[str] = None,
                                 image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepares the request payload specific to Ollama's /api/generate endpoint format.
        Request format:
        {
          "model": "model_name",
          "prompt": "full_prompt_string",
          "stream": false,
          "keep-alive": 30
          "images": ["base64_image_string"] (optional)
        }
        """
        # Construct the full prompt string
        full_prompt_parts = []
        if system_prompt:
            full_prompt_parts.append(system_prompt)
        full_prompt_parts.append(user_message)
        full_prompt = "\n\n".join(full_prompt_parts) if system_prompt else user_message

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "keep-alive": 30
        }

        if image_path:
            try:
                logger.debug(f"Attempting to encode image at path: {image_path} for Ollama /api/generate style payload")
                base64_image = encode_image_to_base64(image_path)
                payload["images"] = [base64_image]
                logger.debug(f"Added base64 image to 'images' field for: {image_path}")
            except FileNotFoundError:
                logger.error(f"File not found at path '{image_path}' for Ollama payload.")
                raise
            except Exception as e:
                logger.error(f"Error encoding image at path '{image_path}' for Ollama: {e}")
                raise RuntimeError(f"Failed to process image at {image_path} for Ollama payload.")

        logger.debug(f"Prepared Ollama (/api/generate style) request payload: {json.dumps(payload, indent=2)}")
        return payload

    def extract_response_message(self, response_json: Dict[str, Any]) -> str:
        """
        Extracts the assistant's message from the Ollama response.
        Refers to: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        """
        try:
            content = response_json["response"]
            return str(content)
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error extracting message from Ollama response: {e}. Full response: {response_json}", exc_info=True)
            raise
