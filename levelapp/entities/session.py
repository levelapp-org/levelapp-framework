"""levelapp/entities/session.py"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime

from levelapp.entities.test_case import TestCase


class EvaluationSession(BaseModel):
    """Represents an evaluation session for tracking a run."""
    session_id: str = str(uuid4())
    timestamp: datetime = datetime.now()
    config: Dict[str, Any]
    test_cases: List[TestCase] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the session."""
        self.test_cases.append(test_case)

    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a comparison result to the session"""
        self.results.extend(result)
        