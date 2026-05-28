from __future__ import annotations

from enum import Enum
from typing import Any


class ImputePhase(str, Enum):
    NONE = "none"
    PRE_FEATURE = "pre_feature"
    POST_FEATURE_REMOVE_ROWS = "post_feature_remove_rows"
    POST_FEATURE_KNN = "post_feature_knn"

    @classmethod
    def parse(cls, value: Any) -> "ImputePhase":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.PRE_FEATURE
        s = str(value).lower()
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"Unknown impute_phase: {value}. Expected one of {[m.value for m in cls]}")
