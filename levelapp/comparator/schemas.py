"""'comparator/schemas.py': Defines Pydantic models for extracted metadata."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, utils


class AttrCompMixin:
    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        attr_name = next(iter(self.__dict__.keys()))
        _cond = (
            fuzz.ratio(
                s1=getattr(self, attr_name),
                s2=getattr(other, attr_name),
                processor=utils.default_process,
            )
            > 99
        )
        return _cond


class CompScoreMixin:
    def comp_score(self, other) -> float:
        attr_name = next(iter(self.__dict__.keys()))
        _score = fuzz.ratio(
            s1=getattr(self, attr_name),
            s2=getattr(other, attr_name),
            processor=utils.default_process,
        )
        return _score


class EntityMetric(str, Enum):
    WRATIO = "wratio"
    LEV_NORM = "lev-norm"
    JARO_WINKLER = "jaro-winkler"
    TOKEN_SORT_RATIO = "token-sort-ratio"
    TOKEN_SET_RATIO = "token-set-ratio"

    @classmethod
    def list(cls):
        return list(map(lambda x: x.value, cls))


class SetMetric(str, Enum):
    ACCURACY = "accuracy"
    F1_SCORE = "f1-score"


class MetricConfig(BaseModel):
    """
    Configuration for a field's comparison metric.
    """
    field_name: str = Field(..., description="Name of the field")
    entity_metric: EntityMetric = Field(..., description="Entity level metric")
    set_metric: Optional[SetMetric] = Field(default=SetMetric.ACCURACY, description="Set level metric")
    threshold: float = Field(..., ge=0, le=100, description="Match threshold")
