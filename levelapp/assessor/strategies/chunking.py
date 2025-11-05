"""levelapp/assessor/strategies/chunking.py"""
from typing import List, Dict, Any

from levelapp.assessor.registry import ChunkingStrategy
from levelapp.assessor.schemas import Document


class SimpleChunkingStrategy(ChunkingStrategy):
    """Splits documents text into naive fixed-size chunks by sentence length."""
    async def run(self, document: Document) -> List[Dict[str, Any]]:
        text = document.content.strip()
        chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]

        return [
            {
                "text": chunk,
                "parent_id": document.id,
                "source_type": "SimpleChunkingStrategy",
            }
            for chunk in chunks
        ]