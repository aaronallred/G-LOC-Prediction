from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .advanced_data_pipeline import AdvancedDataPipeline
from .traditional_data_pipeline import TraditionalDataPipeline

from .model_type import ModelType
from .models.base import BaseModel

PipelineKind = Literal["auto", "advanced", "traditional"]

@dataclass(frozen=True)
class ExperimentMetadata:
    trial_id: np.ndarray
    trial_ints: np.ndarray
    time_s: np.ndarray
    event_validated: np.ndarray
    subject: np.ndarray
    afe_indicator: np.ndarray

class DataPipeline:
    """Facade that routes data loading to the advanced or traditional backend.

    This class is the single entry point for data preparation. It selects a backend
    either explicitly via ``pipeline_kind`` or automatically from ``model.is_traditional``.
    """

    def __init__(
            self,
            data_path: str = "../data/",
            model: Optional[BaseModel] = None,
            model_type: Optional[ModelType] = None,
            random_seed: int = 42,
            num_splits: int = 5,
            kfold_ID: int = 0,
            impute_root: str = "cached_data",
    ) -> None:
        self.data_path = data_path
        self.model = model
        self.random_seed = random_seed
        self.num_splits = num_splits
        self.kfold_ID = kfold_ID
        self.impute_root = impute_root

        self._model_type = model_type
        self._backend: Optional[Any] = None

    @property
    def model_type(self) -> Optional[ModelType]:
        return self._model_type

    @property
    def backend(self) -> Any:
        if self._backend is None:
            self._backend = self._build_backend()
        return self._backend

    def get_data(self, **kwargs: Any) -> Any:
        """Execute the selected backend data pipeline.

        For advanced pipelines this returns:
        ``x_train, x_test, y_train, y_test, all_features``

        For traditional pipelines this returns:
        ``x_feature_matrix, y_gloc_labels``
        """
        backend_type = self._resolve_pipeline_kind()
        request_kwargs = dict(kwargs)

        if self._model_type is not None and "model_type" not in request_kwargs:
            request_kwargs["model_type"] = self._model_type

        if backend_type == "advanced":
            request_kwargs.setdefault("num_splits", self.num_splits)
            request_kwargs.setdefault("kfold_ID", self.kfold_ID)
            request_kwargs.setdefault("impute_path", self._default_impute_path())
            request_kwargs.setdefault("impute_type", 1)
        else:
            request_kwargs.setdefault("classifier_type", self._resolve_classifier_name())
            request_kwargs.setdefault("select_features", self._resolve_select_features(request_kwargs))
            request_kwargs.setdefault("backstep", 0)
            request_kwargs.setdefault("data_rate", 25)
            request_kwargs.setdefault("offset", 0)
            request_kwargs.setdefault("time_start", 0)
            request_kwargs.setdefault("remove_NaN_trials", True)
            request_kwargs.setdefault("subject_to_analyze", None)
            request_kwargs.setdefault("trial_to_analyze", None)
            request_kwargs.setdefault("analysis_type", 2)

        return self.backend.get_data(**request_kwargs)

    def _build_backend(self) -> Any:
        if self.model.is_traditional:
            return TraditionalDataPipeline(
                data_path = self.data_path,
                random_seed = self.random_seed
            )
        else:
            return AdvancedDataPipeline(
                data_path = self.data_path,
                random_seed = self.random_seed
            )

    def _resolve_pipeline_kind(self) -> Literal["advanced", "traditional"]:
        if self.model is None:
            raise ValueError("model is required to resolve pipeline_kind.")

        return "traditional" if self.model.is_traditional else "advanced"

    def _resolve_classifier_name(self) -> str:
        if self.model is None or not hasattr(self.model, "get_name"):
            raise ValueError(
                "Traditional pipeline requires model.get_name() to resolve classifier_type."
            )
        return str(self.model.get_name())

    def _resolve_select_features(self, current_kwargs: dict[str, Any]) -> list[str]:
        if "select_features" in current_kwargs and current_kwargs["select_features"] is not None:
            return list(current_kwargs["select_features"])

        if self.model is None:
            raise ValueError(
                "Traditional pipeline requires select_features. Pass select_features=... "
                "or provide model.config['select_features']."
            )

        config = getattr(self.model, "config", {}) or {}
        features = config.get("select_features")
        if features is None:
            raise ValueError(
                "Traditional pipeline requires select_features. Pass select_features=... "
                "or provide model.config['select_features']."
            )
        return list(features)

    def _default_impute_path(self) -> str:
        model_name = "unknown_model"
        if self.model is not None and hasattr(self.model, "get_name"):
            model_name = str(self.model.get_name())

        afe_filter = self._model_type.afe_filter if self._model_type is not None else "unknown"
        feature_set = self._model_type.feature_set if self._model_type is not None else "unknown"
        filename = f"imputed_{afe_filter}_{feature_set}_fold{self.kfold_ID}.pkl"
        return str(Path(self.impute_root) / model_name / filename)