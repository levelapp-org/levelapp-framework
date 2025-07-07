"""levelapp/entities/test_case.py"""
from typing import Any, Optional

from pydantic import BaseModel


class TestCase(BaseModel):
    """Represents a single test case for evaluation"""
    input_data: Any
    expected_output: Any
    actual_output: Optional[Any]
    # TODO: see whether keep these two attributes or not.
    preprocessed_input: Optional[Any] = None
    preprocessed_expected: Optional[Any] = None