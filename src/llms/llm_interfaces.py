from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import logging
import requests
import json

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract Base Class for LLM providers.
    Defines the contract for sending requests and extracting responses.
    """

    def __init__(self, api_url: str, api_key: str):
        """
        Initializes the LLM provider.

        Args:
            api_url: The API endpoint URL for the provider.
            api_key: The API key for authentication.
        """
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

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
