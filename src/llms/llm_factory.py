from typing import Optional
from .llm_interfaces import LLMProvider
from .llm_providers import OpenRouterProvider


class LLMProviderFactory:
    """
    Factory class to create instances of LLM providers.
    """

    @staticmethod
    def create_provider(provider_name: str,
                        model: str,
                        api_url: str,
                        api_key: str,
                        system_prompt: Optional[str] = None) -> LLMProvider:
        """
        Creates and returns an instance of the specified LLM provider.

        Args:
            provider_name: The name of the provider (e.g., "openrouter").
            model: The LLM model name.
            api_url: The API URL for the provider.
            api_key: The API key.
            system_prompt: Optional system prompt.

        Returns:
            An instance of a class that implements LLMProvider.

        Raises:
            ValueError: If the provider_name is not supported.
        """
        if provider_name.lower() == "openrouter":
            return OpenRouterProvider(model=model, api_url=api_url, api_key=api_key, system_prompt=system_prompt)
        # elif provider_name.lower() == "ollama":
        #     return AnotherProvider(model=model, api_url=api_url, api_key=api_key, system_prompt=system_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
