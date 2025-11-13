"""levelapp/assessor/strategies/generator.py"""
import asyncio
from typing import List

from levelapp.assessor.registry import GenerationStrategy
from levelapp.assessor.schemas import Document


class MockGenerationStrategy(GenerationStrategy):
    async def run(self, query: str, retrieved_docs: List[Document]) -> str:
        context = "\n".join([doc.content for doc in retrieved_docs])
        generated = f"Q: {query}\n\nContext:\n{context}\n\nAnswer: This is a synthesized answer."
        await asyncio.sleep(1)
        return generated
