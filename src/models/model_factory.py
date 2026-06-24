from typing import Any, Dict, Optional, Type

from .base import BaseModel
from .extreme_gradient_boosting import ExtremeGradientBoostingModel
from .k_nearest_neighbors import KNearestNeighborsModel
from .linear_discriminant_analysis import LinearDiscriminantAnalysisModel
from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .support_vector_machine import SupportVectorMachineModel
from .lstm import LSTMModel
from .logistic_regression_ts import LogRegTSModel
from .tcn import TCNModel
from .transformer import TransformerModel



class ModelFactory:
	MODEL_FACTORIES_BY_NAME: Dict[str, Type[BaseModel]] = {
		"LogReg": LogisticRegressionModel,
		"RF": RandomForestModel,
		"LDA": LinearDiscriminantAnalysisModel,
		"SVM": SupportVectorMachineModel,
		"EGB": ExtremeGradientBoostingModel,
		"KNN": KNearestNeighborsModel,
		"LSTM": LSTMModel,
		"LogRegTS": LogRegTSModel,
		"TCN": TCNModel,
		"Trans": TransformerModel
	}

	@classmethod
	def create_model(
		cls,
		model_name: str,
		model_hyperparameters: Optional[Dict[str, Any]] = None,
	) -> BaseModel:
		model_factory = cls.MODEL_FACTORIES_BY_NAME.get(model_name)
		if model_factory is None:
			raise ValueError(f"Model '{model_name}' is not supported.")
		
		return model_factory(model_hyperparameters)