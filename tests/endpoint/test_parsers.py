import os
import pytest

from levelapp.endpoint.schemas import RequestSchemaConfig, ResponseMappingConfig
from levelapp.endpoint.parsers import RequestPayloadBuilder, ResponseDataExtractor


@pytest.fixture
def builder():
    return RequestPayloadBuilder()


def test_static_value(builder):
    schema = [
        RequestSchemaConfig(field_path="user.type", value="admin", value_type="static")
    ]
    payload = builder.build(schema=schema, context={})
    assert payload == {"user": {"type": "admin"}}


def test_dynamic_value(builder):
    schema = [
        RequestSchemaConfig(field_path="user.name", value="username", value_type="dynamic")
    ]
    context = {"username": "Sof"}
    payload = builder.build(schema=schema, context=context)
    assert payload == {"user": {"name": "Sof"}}


def test_env_value(builder):
    os.environ["ENV_TEST"] = "staging"
    schema = [
        RequestSchemaConfig(field_path="metadata.env", value="ENV_TEST", value_type="env")
    ]
    payload = builder.build(schema=schema, context={})
    assert payload == {"metadata": {"env": "staging"}}


def test_nested_structure(builder):
    schema = [
        RequestSchemaConfig(field_path="a.b.c", value="nested", value_type="static")
    ]
    payload = builder.build(schema=schema, context={})
    assert payload == {"a": {"b": {"c": "nested"}}}


def test_missing_required_value(builder):
    schema = [
        RequestSchemaConfig(field_path="user.name", value="username", value_type="dynamic", required=True)
    ]
    context = {}
    with pytest.raises(ValueError):
        builder.build(schema=schema, context=context)


@pytest.fixture
def extractor():
    return ResponseDataExtractor()


def test_simple_extraction(extractor):
    response = {"json": {"message": "Wech frer!"}}
    mappings = [
        ResponseMappingConfig(field_path="json.message", extract_as="reply")
    ]
    result = extractor.extract(response_data=response, mappings=mappings)
    assert result == {"reply": "Wech frer!"}


def test_missing_field_default(extractor):
    response = {"json": {}}
    mappings = [
        ResponseMappingConfig(field_path="json.message", extract_as="reply", default="?")
    ]
    result = extractor.extract(response_data=response, mappings=mappings)
    assert result == {"reply": "?"}


def test_nested_extraction(extractor):
    response = {"data": {"user": {"id": 42, "name": "Sof"}}}
    mappings = [
        ResponseMappingConfig(field_path="data.user.id", extract_as="user_id"),
        ResponseMappingConfig(field_path="data.user.name", extract_as="user_name"),
    ]
    result = extractor.extract(response_data=response, mappings=mappings)
    assert result == {"user_id": 42, "user_name": "Sof"}


def test_list_index_extraction(extractor):
    response = {"items": [{"id": 1}, {"id": 2}]}
    mappings = [
        ResponseMappingConfig(field_path="items[1].id", extract_as="second_item_id")
    ]
    result = extractor.extract(response_data=response, mappings=mappings)
    assert result == {"second_item_id": 2}


def test_invalid_path(extractor):
    response = {"data": {}}
    mappings = [
        ResponseMappingConfig(field_path="data.missing.key", extract_as="val")
    ]
    result = extractor.extract(response_data=response, mappings=mappings)
    assert result["val"] == None

