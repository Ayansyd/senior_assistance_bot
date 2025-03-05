# llm_inference.py
import requests
import json

class OllamaLLM:
    def __init__(self, model_name: str, api_url: str = "http://localhost:11434"):
        """
        :param model_name: Name of the model as registered in Ollama (e.g., "llama3.2:latest")
        :param api_url: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.api_url = api_url.rstrip('/')  # Ensure no trailing slash

    def generate_response(self, prompt: str, max_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Sends a prompt to the Ollama model and returns the generated response.

        :param prompt: The input text prompt
        :param max_tokens: Maximum number of tokens to generate
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling parameter
        :return: Generated text response
        """
        endpoint = f"{self.api_url}/api/generate"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False  # Set to True if you want to handle streaming responses
        }

        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raises HTTPError, if one occurred.
            result = response.json()
            generated_text = result.get('response', '').strip()
            if not generated_text:
                raise ValueError("No 'response' field found in the API response.")
            return generated_text
        except requests.exceptions.RequestException as e:
            print(f"[OllamaLLM Error]: {e}")
            return "[Sorry, I encountered an error while generating a response.]"
        except (KeyError, IndexError, ValueError) as e:
            print(f"[OllamaLLM Parsing Error]: {e}")
            return "[Sorry, I couldn't parse the response correctly.]"
