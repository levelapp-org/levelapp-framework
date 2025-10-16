import json

import pytest
from pydantic import BaseModel

from levelapp.repository.filesystem import FileSystemRepository


class MockModel(BaseModel):
    id: str
    name: str
    value: int


class MockRepositoryConfig(BaseModel):
    base_path: str


class MockWorkflowConfig(BaseModel):
    repository: MockRepositoryConfig


@pytest.fixture
def temp_repo(tmp_path):
    """Fixture providing a temporary FileSystemRepository instance."""
    config = MockWorkflowConfig(repository=MockRepositoryConfig(base_path=str(tmp_path)))
    repo = FileSystemRepository(config=config)
    yield repo

    for item in tmp_path.glob("**/*"):
        if item.is_file():
            item.unlink()


def test_store_and_retrieve_document(temp_repo):
    model = MockModel(id="1", name="alpha", value=10)

    temp_repo.store_document("user1", "sectionA", "data", "doc1", model)
    retrieved = temp_repo.retrieve_document(
        "user1",
        "sectionA",
        "data",
        "doc1",
        MockModel
    )

    assert retrieved == model
    assert retrieved.name == "alpha"
    assert retrieved.value == 10


def test_query_collection(temp_repo):
    models = [
        MockModel(id="1", name="alpha", value=10),
        MockModel(id="2", name="beta", value=20),
        MockModel(id="3", name="alpha", value=30),
    ]

    for m in models:
        temp_repo.store_document("user1", "sectionA", "data", m.id, m)

    results = temp_repo.query_collection(
        "user1",
        "sectionA",
        "data",
        {"name": "alpha"},
        MockModel
    )

    assert len(results) == 2
    assert all(r.name == "alpha" for r in results)


def test_delete_document(temp_repo):
    model = MockModel(id="1", name="gamma", value=99)
    temp_repo.store_document("user1", "sectionA", "data", "doc1", model)

    deleted = temp_repo.delete_document("user1", "sectionA", "data", "doc1")
    assert deleted is True

    path = temp_repo._compose_path("user1", "sectionA", "data", "doc1")
    assert not path.exists()


def test_retrieve_nonexistent_document(temp_repo):
    retrieved = temp_repo.retrieve_document("u1", "s1", "sub", "missing", MockModel)
    assert retrieved is None


def test_query_empty_collection(temp_repo):
    results = temp_repo.query_collection("userX", "secY", "data", {"name": "none"}, MockModel)
    assert results == []


def test_store_document_creates_directories(temp_repo):
    model = MockModel(id="1", name="delta", value=5)
    temp_repo.store_document("user2", "nested", "deep", "docX", model)

    path = temp_repo._compose_path("user2", "nested", "deep", "docX")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["name"] == "delta"
