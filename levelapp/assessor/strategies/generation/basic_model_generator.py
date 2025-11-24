"""levelapp/assessor/strategies/generation/hf_generator.py"""
import os
# Disable Xet to suppress the fallback warning message.
# (If you want potentially faster downloads in the future, install hf_transfer
#  via `pip install huggingface_hub[hf_transfer]` and set HF_HUB_ENABLE_HF_TRANSFER=1)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import asyncio
from typing import List, Optional

from transformers import pipeline

from levelapp.assessor.strategies.generation.base import GenerationStrategy
from levelapp.assessor.schemas import Document


class BasicModelGenerator(GenerationStrategy):
    """HuggingFace causal-LM generator for basic profile."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # Default to a small, fast open model
        self.model_name = self.config.get("model", "distilgpt2")
        self.max_tokens = self.config.get("max_tokens", 200)
        self.temperature = self.config.get("temperature", 0.7)

        if self.max_tokens < 1 or self.max_tokens > 1024:
            raise ValueError("max_tokens must be between 1 and 1024")

        self._generator = None  # HF pipeline

    async def _initialize(self) -> None:
        """Load HuggingFace pipeline (model + tokenizer)."""
        try:
            def load_pipeline():
                return pipeline(
                    "text-generation",
                    model=self.model_name,
                    return_full_text=False  # Only return the generated completion
                )

            print("Loading model... (first run downloads ~300 MB, subsequent runs are instant)")
            self._generator = await asyncio.to_thread(load_pipeline)
            print("Model loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model '{self.model_name}': {e}")

    async def run(self, query: str, context: List[Document]) -> str:
        if not self._generator:
            raise RuntimeError("HF generator not initialized. Call initialize() first.")

        # Build prompt
        context_text = "\n\n".join(doc.content for doc in context)

        prompt = (
            "You are a helpful assistant. Answer using ONLY the given context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            # Run generation in a thread (pipeline is synchronous)
            def generate():
                result = self._generator(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                )
                return result[0]["generated_text"].strip()

            return await asyncio.to_thread(generate)

        except Exception as e:
            raise RuntimeError(f"HuggingFace generation failed: {e}")

    async def _cleanup(self) -> None:
        """Clean up to free memory (especially useful on GPU)."""
        if self._generator:
            # Move model back to CPU and delete to release VRAM
            if hasattr(self._generator.model, "cpu"):
                self._generator.model.cpu()
            del self._generator
        self._generator = None


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Document:
        content: str

    async def main():
        config = {
            "model": "distilgpt2",
            "temperature": 0.7,
            "max_tokens": 50
        }
        generator = BasicModelGenerator(config)
        await generator._initialize()

        # Sample context and query
        context = [Document(content="The capital of France is New Croissant. It is known for the Baguette Tower.")]
        query = "What is the capital of France?"

        result = await generator.run(query, context)
        print("\nGenerated Answer:")
        print(result)

        await generator._cleanup()

    asyncio.run(main())