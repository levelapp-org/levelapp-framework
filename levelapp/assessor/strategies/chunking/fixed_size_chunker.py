""" levelapp/assessor/strategies/chunking/fixed_size_chunker.py"""
from typing import List

from levelapp.assessor.strategies.chunking.base import ChunkingStrategy
from levelapp.assessor.schemas import Document


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking strategy for basic profile."""
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 300)
        self.overlap = self.config.get("overlap", 50)

        # Validate configuration
        if self.chunk_size < 50 or self.chunk_size > 2000:
            raise ValueError("Chunk size must be between 50 and 2000.")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be between 0 and the chunk size.")

    async def _initialize(self) -> None:
        # No heavy initialization need for this basic chunker
        pass

    async def run(self, documents: List[Document]) -> List[Document]:
        """Split documents into fixed sized chunks with overlap."""
        chunks = []

        for doc in documents:
            text = doc.content
            start_idx = 0

            while start_idx < len(text):
                # Calculate end index for this chunk
                end_idx = min(start_idx + self.chunk_size, len(text))

                # Extract chunk text
                chunk_text = text[start_idx:end_idx]

                # Create chunk document with metadata
                chunk_doc = Document(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata.get('doc_id', 'doc')}_{len(chunks)}",
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "chunk_size": len(chunk_text),
                        "overlap": self.overlap if start_idx > 0 else 0,
                    }
                )

                chunks.append(chunk_doc)

                # Move to next chunk position
                start_idx += self.chunk_size - self.overlap

                if start_idx >= len(text):
                    break

        return chunks
    