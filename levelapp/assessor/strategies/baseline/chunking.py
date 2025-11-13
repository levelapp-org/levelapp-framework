"""levelapp/assessor/strategies/chunking.py"""
import nltk

from typing import List, Dict, Any

from levelapp.assessor.registry import ChunkingStrategy
from levelapp.assessor.schemas import Document


nltk.download('punkt', quiet=True)


class BaselineChunkingStrategy(ChunkingStrategy):
    """Split document content into sentence-base chunks."""
    async def run(self, document: Document) -> List[Dict[str, Any]]:
        sentences = nltk.sent_tokenize(document.content.strip())
        chunks = []

        for idx, sentence in enumerate(sentences):
            chunks.append(
                {
                    "id": f"{document.id}_chunk_{idx}",
                    "text": sentence,
                    "parent_id": document.id,
                    "source_type": str(self.__class__.__name__),
                }
            )

        return chunks
