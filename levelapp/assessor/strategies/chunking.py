"""levelapp/assessor/strategies/chunking.py"""
import asyncio

from typing import List

from levelapp.assessor.registry import ChunkingStrategy
from levelapp.assessor.schemas import Document


class SimpleChunkingStrategy(ChunkingStrategy):
    """Splits documents text into naive fixed-size chunks by sentence length."""
    async def run(self, document: Document) -> List[str]:
        max_len = self.config.get("chunk_size", 300)
        text = document.content
        words = text.split()
        chunks = [
            " ".join(words[i:i + max_len])
            for i in range(0, len(words), max_len)
        ]

        await asyncio.sleep(0)
        return chunks
