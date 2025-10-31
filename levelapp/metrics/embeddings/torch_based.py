"""levelapp/metrics/embeddings/torch_based.py"""
import torch

from typing import Any, Dict
from transformers import AutoTokenizer, AutoModel

from levelapp.core.base import BaseMetric


class TorchEmbeddingMetric(BaseMetric):
    """Embedding similarity using a Transformer model (mean-pooled embeddings)."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        super().__init__(processor=kwargs.get("processor"), score_cutoff=kwargs.get("score_cutoff"))
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def compute(self, generated: str, reference: str) -> Dict[str, Any]:
        self._validate_inputs(generated=generated, reference=reference)

        encoded_input = self.tokenizer(
            [reference, generated],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        model_output = self.model(**encoded_input)

        # Mean pooling
        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

        return {
            "similarity": similarity,
            "metadata": self._build_metadata(
                backend="torch",
                model=self.model_name,
                device=str(self.device),
            )
        }

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
