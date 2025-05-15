from core.llm_providers import LLMProvider, OllamaProvider
from typing import Optional
import requests


def is_ollama_available(ollama_url: str) -> bool:
    """
    Checks if the Ollama service is available at the given URL.
    """
    try:
        # Ollama's root endpoint usually returns "Ollama is running" with a 200 OK.
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            print(f"Ollama is available at {ollama_url}. Response: {response.text[:100]}")
            return True
        else:
            print(f"Ollama returned status {response.status_code} at {ollama_url}.")
            return False
    except Exception as e:
        print(f"An unexpected error occurred while checking Ollama availability: {e}")
        return False


def get_ollama_text_completion(
    user_prompt: str,
    model_name: str = "gemma3:4b-it-q8_0",
    system_prompt: Optional[str] = None,
    api_url: str = None
) -> Optional[str]:
    """
    Gets a text completion from an Ollama model using the /api/generate endpoint.

    Args:
        user_prompt: The user's message or prompt.
        model_name: The name of the Ollama model to use (e.g., "llama3:latest").
        system_prompt: An optional system message to guide the LLM's behavior.
        api_url: The API endpoint for Ollama text generation.

    Returns:
        The LLM's text response, or None if an error occurs.
    """
    assert api_url
    try:
        # For text-only, image_model can be a placeholder or same as text_model
        # api_key is empty for default local Ollama.
        provider: LLMProvider = OllamaProvider(
            image_model=model_name,
            text_model=model_name,
            api_url=api_url,
            api_key=""
        )
        print(f"Requesting completion from model '{model_name}' with prompt: '{user_prompt}'")
        if system_prompt:
             print(f"Using system prompt: '{system_prompt}'")

        completion = provider.get_completion(
            model=model_name,
            user_message=user_prompt,
            system_prompt=system_prompt,
            image_path=None
        )
        return completion
    except Exception as e:
        print(f"Failed to get text completion from Ollama model '{model_name}': {e}", exc_info=True)
        return None


if __name__ == "__main__":

    print("--- Ollama Test Script ---")

    # 1. Check if Ollama is available
    print("\n1. Checking Ollama availability...")
    if is_ollama_available("http://127.0.0.1:11434"):
        print("Ollama instance is available!")

        # 2. Get a text completion
        print("2. Requesting text completion from model: ...")
        my_prompt = "Explain the concept of recursion in one sentence."
        # my_system_prompt = "You are a very concise assistant."
        my_system_prompt = None

        completion_result = get_ollama_text_completion(
            user_prompt=my_prompt,
            api_url="http://127.0.0.1:11434/api/generate"
        )

        if completion_result:
            print(f"\nUser Prompt: {my_prompt}")
            if my_system_prompt:
                print(f"System Prompt: {my_system_prompt}")
            print(f"Ollama Response: {completion_result}")
        else:
            print("Failed to get a completion from Ollama.")
            print("Please ensure Ollama is running and the model is pulled (e.g., `ollama pull `).")

    else:
        print("Ollama instance is NOT available. Please ensure Ollama is running.")

    print("\n--- Test Script Finished ---")
