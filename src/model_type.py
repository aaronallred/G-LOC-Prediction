from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen = True)
class ModelType:
    afe_filter: str  # "noAFE" or "Complete"
    feature_set: str # "Explicit" or "Implicit"

    def get_folder_name(self) -> str:
        return f"{self.afe_filter}_{self.feature_set}"