import os

import ollama
from google import genai


class LLMService:
    """Generate answers using Ollama locally or Gemini in the cloud."""

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
    ):
        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or self._get_streamlit_secret("LLM_PROVIDER")
            or "ollama"
        ).strip().lower()

        self.model_name = (
            model_name
            or os.getenv("LLM_MODEL")
            or self._get_streamlit_secret("LLM_MODEL")
            or "llama3.2:3b"
        )

        if self.provider == "gemini":
            api_key = (
                os.getenv("GEMINI_API_KEY")
                or self._get_streamlit_secret("GEMINI_API_KEY")
            )

            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required when using Gemini."
                )

            self.client = genai.Client(api_key=api_key)

        elif self.provider != "ollama":
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}"
            )

    @staticmethod
    def _get_streamlit_secret(name: str) -> str | None:
        """
        Read a secret from Streamlit when running inside Streamlit.

        Returns None when Streamlit is unavailable or the secret
        does not exist.
        """
        try:
            import streamlit as st

            value = st.secrets.get(name)

            if value is None:
                return None

            return str(value)

        except Exception:
            return None

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from a prepared prompt.

        Args:
            prompt: Complete prompt prepared by PromptBuilder.

        Returns:
            Generated answer.

        Raises:
            ValueError: If the prompt is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        if self.provider == "gemini":
            interaction = self.client.interactions.create(
                model=self.model_name,
                input=prompt,
            )

            return (interaction.output_text or "").strip()

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"].strip()