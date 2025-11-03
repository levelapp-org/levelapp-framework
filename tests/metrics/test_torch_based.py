import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from levelapp.metrics.embeddings.torch_based import TorchEmbeddingMetric


@pytest.fixture(scope="module")
def torch_metric():
    return TorchEmbeddingMetric(model_name="sentence-transformers/all-MiniLM-L6-v2")


def test_mean_pooling_shape(torch_metric):
    """Ensure mean pooling outputs correct shape."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = torch_metric.tokenizer
    model = torch_metric.model
    input_text = ["Tony Soprano"]
    encoded = tokenizer(input_text, return_tensors="pt", padding=True)

    with torch.no_grad():
        output = model(**encoded)

    pooled = torch_metric._mean_pooling(output, encoded["attention_mask"])

    assert pooled.shape[0] == len(input_text)
    assert pooled.shape[1] == output[0].shape[-1]


def test_similarity_self_high(torch_metric):
    """Self-similarity should be high."""
    result = torch_metric.compute("Oh, shit! Here we go again", "Oh, shit! Here we go again")
    assert result["similarity"] > 0.0


def test_similarity_diff_low(torch_metric):
    """Similarity between unrelated sentences should be lower."""
    result = torch_metric.compute(
        "All this from a slice of gabagool?",
        "Everything we see and experience is not all there is."
    )
    assert result["similarity"] < 0.8
