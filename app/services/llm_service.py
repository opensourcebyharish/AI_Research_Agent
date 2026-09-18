import ollama


class LLMService:
    """Generate answers using a local Ollama language model."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
    ):
        self.model_name = model_name

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
            raise ValueError(
                "prompt must not be empty"
            )

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
    