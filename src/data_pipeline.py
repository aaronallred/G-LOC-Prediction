from abc import ABC, abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import json
import logging
import os
import pickle
import re

import faiss
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from baseline import BaselineContext, baseline_data
from features import FEATURE_REGISTRY, RawEEGGroup, ProcessedEEGGroup
from GLOC_experiment_config_parser import GLOCExperimentConfigParser
from model_type import ModelType
from models.base import BaseModel

logger = logging.getLogger(__name__)
SOURCE_DIR = Path(__file__).resolve().parent


def _resolve_from_source_dir(path_value: str) -> str:
    """Resolve relative paths against this module's directory."""
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((SOURCE_DIR / candidate).resolve())

class DataPipeline:
    """Facade that routes data loading to the advanced or traditional backend.

    This class is the single entry point for data preparation. It selects a backend
    either explicitly based on the model type and routes to the appropriate implementation.
    
    Configuration is sourced from GLOCExperimentConfigParser for reproducibility and
    centralized configuration management.
    
    Args:
        config_parser: GLOCExperimentConfigParser instance containing experiment configuration
        model: Optional BaseModel instance for type resolution
        data_path_override: Optional override for data_path from config (for testing)
    """

    def __init__(
            self,
            config_parser: GLOCExperimentConfigParser,
    ) -> None:
        """Initialize the facade with the experiment configuration parser."""
        self._config_parser = config_parser

    _SENSOR_STREAM_PATTERNS: dict[str, tuple[str, ...]] = {
        # Equivital streams
        "ECG": (r"ecg",),
        "HR": (r"hr", r"participant_hr"),
        "BR": (r"br",),
        "Temperature": (r"temp", r"temperature"),
        # Other device streams
        "Pupil": (r"pupil",),
        "Centrifuge": (r"centrifuge",),
        "EEG": (r"eeg",),
        "Strain": (r"strain",),
        # Demographics
        "Participant": (r"participant_",),
        "Demographics": (r"participant_",),
    }

    def get_data(self) -> Any:
        """Execute the selected backend data pipeline.

        For advanced pipelines this returns:
        ``x_train, x_test, y_train, y_test, all_features``

        For traditional pipelines this returns:
        ``x_feature_matrix, y_gloc_labels``
        
        Returns:
            Tuple or data from the backend pipeline
        """
        backend_type = self._resolve_pipeline_kind()
        backend_data_pipeline = self._build_backend()
        request_kwargs: dict[str, Any] = {
            "model_type": self._config_parser.get_model_type(),
            "remove_NaN_trials": self._config_parser.get_remove_NaN_trials(),
            "subject_to_analyze": self._config_parser.get_subject_to_analyze(),
            "trial_to_analyze": self._config_parser.get_trial_to_analyze(),
            "analysis_type": self._config_parser.get_analysis_type(),
            "output_feature_dtype": self._config_parser.get_output_feature_dtype(),
            "impute_file_name": self._config_parser.get_impute_file_name(),
            "should_impute": self._config_parser.get_should_impute(),
            "save_impute": self._config_parser.get_save_impute(),
            "load_impute": self._config_parser.get_load_impute()
        }

        if backend_type == "advanced":
            request_kwargs["num_splits"] = self._config_parser.get_num_splits()
            request_kwargs["kfold_ID"] = self._config_parser.get_kfold_ID()
            request_kwargs["n_neighbors"] = self._config_parser.get_n_neighbors()
            request_kwargs["baseline_window"] = self._config_parser.get_baseline_window()
        else:
            request_kwargs["classifier_type"] = self._resolve_classifier_name()
            selected_features = self._resolve_select_features(request_kwargs)
            selected_features = self._apply_sensor_ablation(selected_features)
            request_kwargs["select_features"] = selected_features
            request_kwargs["backstep"] = self._config_parser.get_backstep()
            request_kwargs["data_rate"] = self._config_parser.get_data_rate()
            request_kwargs["offset"] = self._config_parser.get_offset()
            request_kwargs["time_start"] = self._config_parser.get_time_start()

        return backend_data_pipeline.get_data(**request_kwargs)

    def _build_backend(self) -> Any:
        """Instantiate the backend pipeline selected by model type."""
        pipeline_kind = self._resolve_pipeline_kind()
        
        if pipeline_kind == "traditional":
            logger.info("Selected traditional data pipeline based on model type.")
            return TraditionalDataPipeline(
                data_path=self._config_parser.get_data_path(),
                random_seed=self._config_parser.get_random_seed()
            )
        else:
            logger.info("Selected advanced data pipeline based on model type.")
            return AdvancedDataPipeline(
                data_path=self._config_parser.get_data_path(),
                random_seed=self._config_parser.get_random_seed()
            )

    def _resolve_pipeline_kind(self) -> Literal["advanced", "traditional"]:
        """Resolve whether the configured model maps to advanced or traditional flow."""
        model = self._config_parser.get_model()

        if model is None or not hasattr(model, "is_traditional"):
            raise ValueError("Model does not have 'is_traditional' attribute. Unable to determine pipeline kind.")

        return "traditional" if model.is_traditional else "advanced"
    

    def _resolve_classifier_name(self) -> str:
        """Resolve classifier name from the configured model."""
        model = self._config_parser.get_model()
        if model is None or not hasattr(model, "get_name"):
            raise ValueError("Unable to determine classifier name.")

        return model.get_name()

    def _resolve_select_features(self, current_kwargs: dict[str, Any]) -> list[str]:
        """Load selected feature names from the median-hyperparameter cache."""

        # Function to load in median hyperparameters from a simple JSON
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_type_string = f"{current_kwargs['model_type'].afe_filter}_{current_kwargs['model_type'].feature_set}"
        json_path = os.path.join(BASE_DIR, 'ModelSave', 'CV', model_type_string, f'median_hyperparameters_{current_kwargs["classifier_type"]}.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        return data['selected_features']

    def _apply_sensor_ablation(self, selected_features: list[str]) -> list[str]:
        """Optionally restrict selected features to configured sensor streams.

        This is a no-op unless shared_data_parameters.sensor_ablation.enabled is true.
        """
        enabled, streams = self._get_sensor_ablation_config()
        if not enabled:
            return selected_features

        requested_streams = [s.strip() for s in streams if s and s.strip()]
        unknown_streams = [s for s in requested_streams if s not in self._SENSOR_STREAM_PATTERNS]
        if unknown_streams:
            supported = ", ".join(sorted(self._SENSOR_STREAM_PATTERNS.keys()))
            raise ValueError(
                f"Unknown sensor_ablation stream(s): {unknown_streams}. Supported streams: {supported}."
            )

        matched_features: list[str] = []
        compiled_patterns = [
            re.compile(pattern, flags=re.IGNORECASE)
            for stream in requested_streams
            for pattern in self._SENSOR_STREAM_PATTERNS[stream]
        ]

        for feature_name in selected_features:
            if any(pattern.search(feature_name) for pattern in compiled_patterns):
                matched_features.append(feature_name)

        if len(matched_features) == 0:
            raise ValueError(
                "Sensor ablation removed all selected features. "
                f"Requested streams={requested_streams}. "
                "Check stream names and selected feature naming conventions."
            )

        logger.info(
            "Applied sensor ablation for streams=%s. Selected features reduced from %d to %d.",
            requested_streams,
            len(selected_features),
            len(matched_features),
        )
        return matched_features

    def _get_sensor_ablation_config(self) -> tuple[bool, list[str]]:
        """Fetch ablation settings if parser provides them; otherwise default to disabled.

        The attribute checks preserve backward compatibility with parser test doubles.
        """
        get_enabled = getattr(self._config_parser, "get_sensor_ablation_enabled", None)
        get_streams = getattr(self._config_parser, "get_sensor_ablation_streams", None)

        if not callable(get_enabled) or not callable(get_streams):
            return False, []

        return bool(get_enabled()), list(get_streams())
    


class BaseGLOCDataPipeline(ABC):
    """Abstract base class for GLOC data pipelines (advanced and traditional).
    
    Contains all shared methods and constants for data loading, processing, and feature engineering.
    Subclasses must implement the abstract get_data() method with pipeline-specific logic.
    """

    # Shared constants: Feature groups by model type
    FEATURE_GROUPS_BY_MODEL_TYPE = {
        ModelType("noAFE", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ModelType("noAFE", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
        ModelType("Complete", "Explicit"): ("ECG", "BR", "temp", "eyetracking", "G", "rawEEG", "processedEEG", "strain", "demographics"),
        ModelType("Complete", "Implicit"): ("ECG", "BR", "temp", "eyetracking", "rawEEG"),
    }
    
    # Mapping of participant -> DC trial numbers for GOR EEG data files
    _EEG_PARTICIPANT_TRIALS = {
        1: [1, 2, 3],  2: [1, 2, 3],  3: [1, 2, 3],  4: [1, 2, 3],  5: [1, 2, 3],
        6: [1, 4, 6],  7: [2, 4, 6],  8: [1, 3],     9: [2, 5, 6],
        10: [2, 4, 5], 11: [1],       12: [1, 5],     13: [1, 3, 6],
    }

    _EEG_BASELINE_BANDS = ["delta", "theta", "alpha", "beta"]

    # 32 raw EEG channel names (without " - EEG" suffix)
    _RAW_EEG_CHANNELS = [
        'F1', 'Fz', 'F3', 'C3', 'C4', 'CP1', 'CP2',
        'T8', 'TP9', 'TP10', 'P7', 'P8', 'AFz', 'AF4',
        'FT9', 'FT10', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4',
        'FC6', 'C5', 'Cz', 'CP5', 'CP6', 'P5', 'P3',
        'P1', 'Pz', 'P4', 'P6',
    ]

    # Unengineered data streams used for feature selection
    _UNENGINEERED_STREAMS = frozenset(
        [
            'HR (bpm) - Equivital',
            'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital',
            'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital',
            'BR (rpm) - Equivital',
            'Skin Temperature - IR Thermometer (°C) - Equivital',
            'Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii',
            'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
            'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii',
            'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii',
            'magnitude - Centrifuge',
            'Strain [0/1]',
            'participant_gender', 'participant_age', 'participant_height',
            'participant_weight', 'participant_BMI', 'participant_blood_volume',
            'participant_SBP_seated', 'participant_SBP_stand', 'participant_SBP_exercise',
            'participant_DBP_seated', 'participant_DBP_stand', 'participant_DBP_exercise',
            'participant_MAP_seated', 'participant_MAP_stand', 'participant_MAP_exercise',
            'participant_HR_seated', 'participant_HR_stand', 'participant_HR_exercise',
            'participant_max_leg_strength', 'participant_largest_leg_circumference',
            'participant_lower_leg_volume', 'participant_skinfolds_chest_avg',
            'participant_skinfolds_abd_avg', 'participant_skinfolds_thigh_avg',
            'participant_skinfolds_midax_avg', 'participant_skinfolds_subscap_avg',
            'participant_skinfolds_tri_avg', 'participant_skinfolds_supra_avg',
            'participant_skinfolds_sum', 'participant_percent_fat', 'participant_leg_length',
            'participant_arm_length', 'participant_midline_neck_length',
            'participant_lateral_neck_length', 'participant_torso_length_post',
            'participant_torso_length_ax', 'participant_head_to_heart', 'participant_head_girth',
            'participant_neck_girth', 'participant_chest_upper_girth', 'participant_chest_under_girth',
            'participant_waist_girth', 'participant_hip_girth', 'participant_thigh_girth',
            'participant_calf_girth', 'participant_biceps_girth_flex', 'participant_biceps_girth_relax',
            'participant_neck_flexion', 'participant_neck_extension', 'participant_neck_right_rotation',
            'participant_neck_left_rotation', 'participant_neck_left_lat_flex',
            'participant_neck_right_lat_flex', 'participant_pred_vo2',
        ]
        + [f'{ch} - EEG' for ch in _RAW_EEG_CHANNELS]
        + [f'{ch}_{band} - EEG' for ch in _RAW_EEG_CHANNELS for band in ["delta", "theta", "alpha", "beta"]]
    )

    def __init__(self, data_path: str = "../data/", random_seed: int = 42) -> None:
        """Initialize shared pipeline state."""
        self.data_path = _resolve_from_source_dir(data_path)
        self._data_locations = None
        self.random_seed = random_seed

    @abstractmethod
    def get_data(self, **kwargs: Any) -> Any:
        """Execute the data pipeline with specified parameters.
        
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Pipeline-specific keyword arguments
            
        Returns:
            Processed data in pipeline-specific format
        """
        pass

    def _get_data_locations(self) -> Dict[str, Any]:
        """Build and cache filesystem paths used by data loading."""
        if self._data_locations is not None:
            return self._data_locations

        eeg_dir = "GLOC_GOR_EEG_data_participants_1-13"
        list_of_eeg_data_file_paths = [
            os.path.join(self.data_path, eeg_dir, f"GLOC_{p:02d}_DC{t}_25Hz_EEG_power_wMAR.xlsx")
            for p, trials in self._EEG_PARTICIPANT_TRIALS.items()
            for t in trials
        ]

        list_of_baseline_eeg_processed_file_paths = [
            os.path.join(self.data_path, f"GLOC_EEG_baseline_{band}_noAFE1.csv")
            for band in self._EEG_BASELINE_BANDS
        ]

        self._data_locations = {
            "main": os.path.join(self.data_path, "all_trials_25_hz_stacked_null_str_filled.csv"),
            "baseline": os.path.join(self.data_path, "ParticipantBaseline.csv"),
            "demographic": os.path.join(self.data_path, "GLOC_Effectiveness_Final.csv"),
            "eeg_list": list_of_eeg_data_file_paths,
            "baseline_eeg_processed_list": list_of_baseline_eeg_processed_file_paths,
        }

        return self._data_locations
    
    def _load_data(self, file_paths: Dict[str, Any], output_feature_dtype: np.dtype = np.dtype(np.float32)) -> pd.DataFrame:
        """Load data from CSV or pickle files. If pickle does not exist, create it from CSV."""
        main_data_pickle_file = file_paths["main"].replace(".csv", ".pkl")

        # Check if pickle exists, if not create it then save it
        if not os.path.isfile(main_data_pickle_file):
            logger.info("Pickle not found at %s. Loading from CSV and caching.", main_data_pickle_file)
            gloc_data = pd.read_csv(file_paths["main"])
            gloc_data.to_pickle(main_data_pickle_file)
        else:
            logger.info("Loading data from pickle at %s.", main_data_pickle_file)
            gloc_data = pd.read_pickle(main_data_pickle_file)
        
        # Add GOR and EEG data from other files
        gloc_data = self._process_EEG_GOR(file_paths["eeg_list"], gloc_data, output_feature_dtype)

        # Adjust AFE condition column always
        gloc_data["condition"] = gloc_data["condition"].map({"N": 0, "AFE": 1})
        gloc_data = gloc_data.rename(columns = {"condition": "AFE_indicator"})

        float64_cols = gloc_data.select_dtypes(include="float64").columns
        if len(float64_cols) > 0:
            gloc_data = gloc_data.astype({col: output_feature_dtype for col in float64_cols}).copy()
        
        # Extracting subject and trial into separate columns
        trial_ids = gloc_data["trial_id"].to_numpy().astype("str")
        trial_ids = np.array(np.char.split(trial_ids, "-").tolist())
        gloc_data["subject"] = trial_ids[:, 0]
        gloc_data["trial"] = trial_ids[:, 1]

        # Decouple from original dataframe to prevent unwanted modifications later on
        return gloc_data

    def _process_EEG_GOR(self, list_of_eeg_data_files: List[str], gloc_data: pd.DataFrame, output_feature_dtype: np.dtype = np.dtype(np.float32)) -> pd.DataFrame:
        """Slot in GOR EEG band power data from xlsx files, replacing NaNs in the main CSV."""
        trial_indices_map = gloc_data.groupby("trial_id", sort=False).indices
        event_validated = gloc_data["event_validated"].to_numpy()
        trial_ids = gloc_data["trial_id"].to_numpy()
        begin_mask = event_validated == "begin GOR"
        begin_idx_map = (
            pd.Series(np.flatnonzero(begin_mask), index=trial_ids[begin_mask])
            .groupby(level = 0, sort=False)
            .first()
            .to_dict()
        )

        band_names = ["delta", "theta", "alpha", "beta"]

        for current_file in list_of_eeg_data_files:
            # Parse trial ID from filename: e.g. "GLOC_01_DC1_..." -> "01-01"
            match = re.search(r'GLOC_(\d{2})_DC(\d+)', os.path.basename(current_file))
            if not match:
                logger.warning("Could not parse trial ID from filename: %s", current_file)
                continue
            corresponding_trial = f"{match.group(1)}-0{match.group(2)}"

            # Read all band sheets and drop the time column
            band_dfs = {
                band: pd.read_excel(current_file, sheet_name=band).iloc[:, :-1]
                for band in band_names
            }

            trial_indices = trial_indices_map.get(corresponding_trial)
            if trial_indices is None:
                logger.warning("Could not find trial %s in data.", corresponding_trial)
                continue

            index_begin_GOR = begin_idx_map.get(corresponding_trial)
            if index_begin_GOR is None:
                logger.warning("Could not find 'begin GOR' for trial %s.", corresponding_trial)
                continue

            start_pos = np.searchsorted(trial_indices, index_begin_GOR)
            n_rows = len(band_dfs["delta"])
            trial_indexer = trial_indices[start_pos : start_pos + n_rows]

            # Build column names and assign values for each band
            column_names = band_dfs["delta"].columns
            for band in band_names:
                cols = [f"{c}_{band} - EEG" for c in column_names]
                gloc_data.loc[trial_indexer, cols] = band_dfs[band].to_numpy(dtype=output_feature_dtype)

        return gloc_data

    def _filter_data_by_analysis_type(
            self,
            analysis_type: int,
            gloc_data: pd.DataFrame,
            subject_to_analyze: Optional[str] = None,
            trial_to_analyze: Optional[str] = None,
    ) -> pd.DataFrame:
        """Analyze only section of gloc_data specified using analysis_type."""
        if analysis_type == 0: # One Trial / One Subject
            mask = (gloc_data["subject"] == subject_to_analyze) & (gloc_data["trial"] == trial_to_analyze)
        elif analysis_type == 1: # All Trials for One Subject
            mask = (gloc_data["subject"] == subject_to_analyze)
        else: # All Trials for All Subjects
            return gloc_data
        
        return gloc_data[mask]

    def _process_and_get_feature_names(
            self,
            gloc_data: pd.DataFrame,
            feature_groups_to_analyze: Sequence[str],
            model_type: ModelType,
            file_names: Dict[str, Any],
            output_feature_dtype: np.dtype = np.dtype(np.float32),
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Process data and extract feature names based on specified feature groups."""
        GROUPS_OF_FEATURE_GROUPS = {
            "Phys": {"ECG", "BR", "temp", "fnirs", "eyetracking", "rawEEG", "processedEEG"},
            "ECG": {"ECG"},
            "EEG": {"processedEEG"} # Adding rawEEG does not change anything (rawEEG ignored during baseline v7 and v8 calculations)
        }

        features = {
            "All": [],
            "Phys": [],
            "ECG": [],
            "EEG": []
        }
        features_all = features["All"]
        features_phys = features["Phys"]
        features_ecg = features["ECG"]
        features_eeg = features["EEG"]

        for group_name in feature_groups_to_analyze:
            if group_name not in FEATURE_REGISTRY:
                logger.warning("Feature group '%s' not recognized. Skipping.", group_name)
                continue

            processor = FEATURE_REGISTRY[group_name]

            # Process data for the feature group
            gloc_data = processor.process(gloc_data, file_names)
            feature_names = processor.get_feature_names(model_type)

            # Adding features to relevant groups
            if group_name in GROUPS_OF_FEATURE_GROUPS["Phys"]:
                features_phys.extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["ECG"]:
                features_ecg.extend(feature_names)

            if group_name in GROUPS_OF_FEATURE_GROUPS["EEG"]:
                features_eeg.extend(feature_names)

            features_all.extend(feature_names)

        return gloc_data, features

    def _label_gloc_events(self, gloc_data: pd.DataFrame) -> np.ndarray:
        """Create a GLOC label vector based on event_validated column.
        
        Labels are 1 between GLOC and Return to Consciousness events.
        """
        event_validated = gloc_data["event_validated"]

        # Find all GLOC and RTC indices, pair them in order, and label between each pair
        gloc_indices = np.where(event_validated.to_numpy() == "GLOC")[0]
        rtc_indices = np.where(event_validated.to_numpy() == "return to consciousness")[0]

        trial_ids = gloc_data["trial_id"].to_numpy()
        gloc_labels = np.zeros(len(gloc_data))

        for i in range(len(gloc_indices)):
            start = gloc_indices[i]
            end = rtc_indices[i]
            if trial_ids[start] == trial_ids[end]:
                gloc_labels[start:end] = 1

        return gloc_labels

    def _afe_subset(self, gloc_data: pd.DataFrame, gloc_labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Remove any trial that contains AFE condition (AFE indicator == 1)."""
        trial_has_afe = gloc_data.groupby(["subject", "trial"])["AFE_indicator"].transform("max")
        keep_mask = trial_has_afe != 1

        gloc_data = gloc_data.loc[keep_mask].reset_index(drop = True)
        gloc_labels = gloc_labels[keep_mask]

        return gloc_data, gloc_labels
    
    def _eeg_specific_imputation(self, gloc_data: pd.DataFrame, features: Dict[str, List[str]]) -> pd.DataFrame:
        """Mean-impute EEG channels that are exclusive to one AFE condition."""
        self._eeg_condition_impute(gloc_data, features, gloc_data["AFE_indicator"])
        return gloc_data

    def _eeg_condition_impute(self, gloc_data: pd.DataFrame, features: Dict[str, List[str]], afe_indicator_column: pd.Series, verbose: bool = False) -> None:
        """Mean-impute condition-specific EEG columns so both AFE/non-AFE have all features. Modifies gloc_data in-place."""
        # Create masks for each condition
        afe_mask = afe_indicator_column == 1
        nonafe_mask = afe_indicator_column == 0

        # Pull columns that need to be imputed for each type
        raw_eeg_feature_names = RawEEGGroup.get_separated_feature_names()
        processed_eeg_feature_names = ProcessedEEGGroup.get_separated_feature_names()
        all_afe_only_cols = raw_eeg_feature_names["AFE Only"] + processed_eeg_feature_names["AFE Only"]
        all_nonafe_only_cols = raw_eeg_feature_names["Non-AFE Only"] + processed_eeg_feature_names["Non-AFE Only"]
        eeg_feature_set = set(features["EEG"])
        afe_only_cols = [col for col in all_afe_only_cols if col in eeg_feature_set]
        nonafe_only_cols = [col for col in all_nonafe_only_cols if col in eeg_feature_set]

        # Mean imputation processing
        if afe_only_cols:
            means = gloc_data.loc[afe_mask, afe_only_cols].mean(skipna = True)
            if verbose:
                missing_counts = gloc_data.loc[nonafe_mask, afe_only_cols].isna().sum()
            gloc_data.loc[nonafe_mask, afe_only_cols] = gloc_data.loc[nonafe_mask, afe_only_cols].fillna(means)

            if verbose:
                for col, n in missing_counts.items():
                    logger.debug("Imputed %d values in '%s' for non-AFE rows.", n, col)

        if nonafe_only_cols:
            means = gloc_data.loc[nonafe_mask, nonafe_only_cols].mean(skipna = True)
            if verbose:
                missing_counts = gloc_data.loc[afe_mask, nonafe_only_cols].isna().sum()
            gloc_data.loc[afe_mask, nonafe_only_cols] = gloc_data.loc[afe_mask, nonafe_only_cols].fillna(means)

            if verbose:
                for col, n in missing_counts.items():
                    logger.debug("Imputed %d values in '%s' for AFE rows.", n, col)

    def _remove_all_nan_trials(
            self,
            gloc_data: pd.DataFrame,
            features: Dict[str, List[str]],
            gloc_labels: np.ndarray,
            verbose: bool = False,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """Remove trials where at least one feature is entirely NaN. Returns NaN proportion table."""
        # All features and subject trial info to be put into a reduced dataframe from gloc_data
        all_features = features["All"]
        all_features_with_ids = all_features + ["subject", "trial"]
        reduced_data_frame = gloc_data[all_features_with_ids]

        nan_flags = reduced_data_frame[all_features].isna()
        group_keys = [reduced_data_frame["subject"], reduced_data_frame["trial"]]
        grouped = nan_flags.groupby(group_keys, sort=False)

        nan_proportion_df = grouped.mean()
        all_nan_cols_df = grouped.all()
        bad_trials = all_nan_cols_df.any(axis=1)

        if verbose and bad_trials.any():
            for (subject, trial), is_bad in bad_trials.items():
                if is_bad:
                    nan_features = all_nan_cols_df.columns[all_nan_cols_df.loc[(subject, trial)]].tolist()
                    logger.info("Subject %s, Trial %s: features entirely NaN → %s", subject, trial, nan_features)

        nan_proportion_df.insert(
            0,
            "subject-trial",
            [f"{subject}-{trial}" for subject, trial in nan_proportion_df.index],
        )
        nan_proportion_df.reset_index(drop=True, inplace=True)

        group_ids = reduced_data_frame.groupby(["subject", "trial"], sort=False).ngroup().to_numpy()
        keep_mask = ~bad_trials.to_numpy()[group_ids]

        rows_to_remove = gloc_data.index[~keep_mask]
        gloc_data.drop(rows_to_remove, inplace=True)
        gloc_data.reset_index(drop=True, inplace=True)

        kept_labels = gloc_labels[keep_mask]
        gloc_labels.resize(kept_labels.shape, refcheck=False)
        gloc_labels[:] = kept_labels

        N = int(bad_trials.shape[0])
        M = int(bad_trials.sum())

        logger.info("%d trials with all NaNs for at least one feature out of %d trials. %d remaining.", M, N, N - M)

        return gloc_data, gloc_labels, nan_proportion_df
    
    def _reduce_memory(
            self,
            gloc_data: pd.DataFrame,
            gloc_labels: np.ndarray,
            features: Dict[str, List[str]],
            output_feature_dtype: np.dtype = np.dtype(np.float32),
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Extract numpy arrays from DataFrame and free the DataFrame to reduce memory usage."""
        trial_id_arr = gloc_data["trial_id"].to_numpy()
        experiment_metadata = {
            "trial_id": trial_id_arr,
            "trial_ints": self._convert_to_unique_ordered_integers(trial_id_arr),
            "Time (s)": gloc_data["Time (s)"].to_numpy(dtype = output_feature_dtype),
            "event_validated": gloc_data["event_validated"].to_numpy(dtype = str),
            "subject": gloc_data["subject"].to_numpy(dtype = str),
            "AFE_indicator": gloc_data["AFE_indicator"].to_numpy(dtype = np.bool_).reshape(-1, 1),
        }

        gloc_data_all_features_numpy = np.asarray(gloc_data[features["All"]].to_numpy(dtype = output_feature_dtype), dtype = output_feature_dtype)
        gloc_labels_numpy = gloc_labels.astype(np.bool_)

        del gloc_data, gloc_labels
        return gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata

    def _convert_to_unique_ordered_integers(self, strings: np.ndarray) -> np.ndarray:
        """Convert strings to 1-based integers preserving first-appearance order."""
        codes, _ = pd.factorize(strings, sort=False)
        return (codes + 1).astype(np.uint8)

    def _get_combined_baseline_data(
            self,
            gloc_data_all_features_imputed_numpy: np.ndarray,
            experiment_metadata: Dict[str, Any],
            baseline_window: float,
            baseline_methods_to_use: List[str],
            features: Dict[str, List[str]],
            file_paths: Dict[str, Any],
            model_type: ModelType,
    ) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, np.ndarray], List[str]]:
        """Compute baselines and return combined outputs plus v0 baseline data/names."""
        participant_baseline = pd.read_csv(file_paths["baseline"])
        participant_baseline_rhr = participant_baseline["resting HR [seated]"][:-1]
        participant_baseline_rhr.index = [f"{i:02d}" for i in range(1, 14)]

        eeg_baseline_data = {}
        for filepath in file_paths["baseline_eeg_processed_list"]:
            df = pd.read_csv(filepath)
            df.index = [f"{i:02d}" for i in range(1, 14)]
            # Extract band name from filename pattern: GLOC_EEG_baseline_{band}_noAFE1.csv
            band = os.path.basename(filepath).split("_")[3]
            eeg_baseline_data[band] = df

        # Build feature-group index arrays using set lookups for O(1) membership
        phys_set, ecg_set, eeg_set = set(features["Phys"]), set(features["ECG"]), set(features["EEG"])
        phys_indices = [i for i, f in enumerate(features["All"]) if f in phys_set]
        ecg_indices = [i for i, f in enumerate(features["All"]) if f in ecg_set]
        eeg_indices = [i for i, f in enumerate(features["All"]) if f in eeg_set]

        context = BaselineContext(
            trial_column=experiment_metadata["trial_id"],
            time_column=experiment_metadata["Time (s)"],
            event_validated_column=experiment_metadata["event_validated"],
            subject_column=experiment_metadata["subject"],
            data_by_features={
                "All": gloc_data_all_features_imputed_numpy,
                "Phys": gloc_data_all_features_imputed_numpy[:, phys_indices],
                "ECG": gloc_data_all_features_imputed_numpy[:, ecg_indices],
                "EEG": gloc_data_all_features_imputed_numpy[:, eeg_indices],
            },
            features=features,
            baseline_window=baseline_window,
            model_type=model_type,
            participant_baseline_data=participant_baseline_rhr,
            eeg_baseline_data=eeg_baseline_data,
        )

        combined_baseline, combined_names, baseline_v0, baseline_names_v0, trial_order = baseline_data(baseline_methods_to_use, context)
        experiment_metadata["trial_order"] = trial_order

        return combined_baseline, combined_names, baseline_v0, baseline_names_v0

    def _remove_constant_columns(
            self,
            x_feature_matrix: np.ndarray,
            select_features: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Remove zero-variance columns from a feature matrix."""
        # Find all constant columns
        constant_columns = np.all(x_feature_matrix == x_feature_matrix[0, :], axis=0)
        keep_columns = ~constant_columns

        # Remove all constant columns from feature matrix and names
        x_feature_matrix = x_feature_matrix[:, keep_columns]
        select_features = [feature for feature, keep in zip(select_features, keep_columns) if keep]

        return x_feature_matrix, select_features

    def _build_faiss_knn_index(
            self,
            d: int,
            M: int = 32,
            efSearch: int = 64,
    ) -> Any:
        """Create a FAISS index using GPU FlatL2 when available, otherwise CPU HNSW."""
        get_num_gpus = getattr(faiss, "get_num_gpus", None)
        gpu_count = 0

        if callable(get_num_gpus):
            try:
                gpu_count = int(get_num_gpus())
            except Exception as exc:
                logger.warning("Unable to query FAISS GPU count; defaulting to CPU HNSW. Error: %s", exc)

        if gpu_count > 0:
            gpu_resources_ctor = getattr(faiss, "StandardGpuResources", None)
            gpu_flat_l2_ctor = getattr(faiss, "GpuIndexFlatL2", None)

            if callable(gpu_resources_ctor) and callable(gpu_flat_l2_ctor):
                try:
                    gpu_resources = gpu_resources_ctor()
                    metric_l2 = getattr(faiss, "METRIC_L2", 1)
                    try:
                        index = gpu_flat_l2_ctor(gpu_resources, d, metric_l2)
                    except TypeError:
                        index = gpu_flat_l2_ctor(gpu_resources, d)

                    logger.info("Using FAISS GPU GpuIndexFlatL2 index for KNN imputation.")
                    return index
                except Exception as exc:
                    logger.warning(
                        "GPU detected but FAISS GPU FlatL2 setup failed; falling back to CPU HNSW. Error: %s",
                        exc,
                    )
            else:
                logger.warning(
                    "GPU detected but FAISS GPU FlatL2 bindings are unavailable; falling back to CPU HNSW."
                )

        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efSearch = efSearch

        # Use fixed RNG seed for deterministic HNSW graph construction on CPU.
        rng = faiss.RandomGenerator(self.random_seed)
        index.hnsw.rng = rng

        logger.info("Using FAISS CPU IndexHNSWFlat index for KNN imputation.")
        return index


class AdvancedDataPipeline(BaseGLOCDataPipeline):
    """
    Advanced data pipeline for GLOC event prediction, refactored from load_and_prepare_data_advanced.
    """

    BASELINING_CHARACTERISTICS_BY_MODEL_TYPE = {
        "noAFE": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
        "Complete": ["v0", "v1", "v2", "v5", "v6"],
    }

    def _get_feature_groups_and_baseline_methods(self, model_type: ModelType) -> Tuple[Sequence[str], List[str]]:
        """Resolve feature groups and baseline methods for advanced pipeline variants."""
        feature_groups_to_analyze = self.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]
        baseline_methods_to_use = self.BASELINING_CHARACTERISTICS_BY_MODEL_TYPE[model_type.afe_filter]
        return feature_groups_to_analyze, baseline_methods_to_use

    def get_data(
            self,
            model_type: ModelType,
            num_splits: int,
            kfold_ID: int,
            impute_file_name: str,
            output_feature_dtype: np.dtype = np.dtype(np.float32),
            subject_to_analyze: Optional[str] = None,
            trial_to_analyze: Optional[str] = None,
            should_impute: bool = True,
            n_neighbors: int = 4,
            baseline_window: float = 32.5,
            analysis_type: int = 2,
            remove_NaN_trials: bool = True,
            save_impute: bool = True,
            load_impute: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load raw data and prepare predictor / target sets for advanced classifiers.

        Parameters:
            model_type: ModelType — e.g. ModelType('Complete', 'Explicit')
            num_splits: Number of K-fold CV splits
            kfold_ID: Which fold to use (0 to num_splits-1)
            impute_file_name: Base file name used in data_path/Processed Data
            output_feature_dtype: Numpy dtype for output feature matrix (e.g., 'float32', 'float64')
            subject_to_analyze: Participant number for single-subject analysis
            trial_to_analyze: Trial number for single-trial analysis
            should_impute: Whether to perform KNN imputation (True = KNN on raw data, False = no imputation)
            n_neighbors: Number of KNN imputation neighbors
            baseline_window: Baseline window duration in seconds
            analysis_type: 2=all data, 1=one participant, 0=one trial
            remove_NaN_trials: Remove trials with all-NaN sensors
            save_impute: Save imputed data to pickle
            load_impute: Load imputed data from pickle if available

        Returns:
            x_train, y_train, x_test, y_test, all_features
        """
        ################################################### FEATURES SETUP ###################################################
        logger.info("Setting up features and baselines for model_type=%s", model_type)
        feature_groups_to_analyze, baseline_methods_to_use = self._get_feature_groups_and_baseline_methods(model_type)

        ############################################# LOAD AND PROCESS DATA #############################################
        logger.info("Loading and processing data with parameters: model_type=%s, subject_to_analyze=%s, trial_to_analyze=%s, analysis_type=%d", model_type, subject_to_analyze, trial_to_analyze, analysis_type)
        file_paths = self._get_data_locations()
        gloc_data = self._load_data(file_paths, output_feature_dtype)
        gloc_data = self._filter_data_by_analysis_type(analysis_type, gloc_data, subject_to_analyze, trial_to_analyze)
        gloc_data, features = self._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths, output_feature_dtype)
        gloc_labels = self._label_gloc_events(gloc_data)
        if model_type.afe_filter != "Complete":
            gloc_data, gloc_labels = self._afe_subset(gloc_data, gloc_labels)

        ############################################# EEG Specific Imputation #############################################
        # logger.info("Performing EEG-specific imputation for model_type=%s", model_type)
        # ####
        # #  Note: This runs for 'complete' models, but because we are only using shared/overlapping EEG features for the
        # #      'complete' case, this block doesn't do anything. Imputation occurs only for non-shared EEG features are used.
        # #       This block requires 'AFE' to be an
        # ####
        # if model_type.afe_filter == "Complete":
        #     gloc_data = self._eeg_specific_imputation(gloc_data, features)

        ############################################### MISSING DATA HANDLING ###############################################
        logger.info("Handling missing data with should_impute=%s, n_neighbors=%d, remove_NaN_trials=%s", should_impute, n_neighbors, remove_NaN_trials)
        # Optional handling of raw NaN data, depending on remove_NaN_trials and should_impute
        if remove_NaN_trials:
            # This also returns a DataFrame with proportion of NaN values for each feature for each trial
            # Also modifies gloc_data and gloc_labels to remove trials with all NaNs in at least one feature
            # Note: DataFrame not used for the pipeline for memory purposes
            gloc_data, gloc_labels, _ = self._remove_all_nan_trials(gloc_data, features, gloc_labels)

        ################################################## REDUCE MEMORY ##################################################
        logger.info("Reducing memory usage by converting to numpy arrays with dtype=%s.", output_feature_dtype)
        gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata = self._reduce_memory(
            gloc_data, gloc_labels, features, output_feature_dtype
        )

        ################################################## Impute Missing ##################################################
        logger.info("Imputing missing data with should_impute=%s, n_neighbors=%d, impute_file_name=%s, save_impute=%s, load_impute=%s", should_impute, n_neighbors, impute_file_name, save_impute, load_impute)
        if should_impute:
            gloc_data_all_features_imputed_numpy = self._impute_missing_data(
                gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata,
                impute_file_name, save_impute, load_impute, num_splits, kfold_ID, n_neighbors
            )
        else:
            logger.info("Skipping KNN imputation as should_impute=False.")
            gloc_data_all_features_imputed_numpy = gloc_data_all_features_numpy

        ################################################## BASELINE DATA ##################################################
        logger.info("Calculating baselines with methods: %s", baseline_methods_to_use)
        combined_baseline, combined_baseline_names, _, _ = self._get_combined_baseline_data(
            gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window,
            baseline_methods_to_use, features, file_paths, model_type
        )

        ################################################ GENERATE FEATURES ################################################
        logger.info("Generating features for model_type=%s", model_type)
        x_feature_matrix, features["All"] = self._generate_features(
            baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata, output_feature_dtype
        )

        ############################################# FEATURE CLEAN AND PREP ##############################################
        logger.info("Cleaning and preparing features for model_type=%s", model_type)
        x_feature_matrix, y_gloc_labels, features["All"], experiment_metadata["trial_ints"] = self._feature_clean_and_prep(
            x_feature_matrix, gloc_labels_numpy, features, experiment_metadata, model_type, should_impute
        )

        ################################################ TRAIN/TEST SPLIT  ################################################
        logger.info("Performing train/test split with num_splits=%d, kfold_ID=%d", num_splits, kfold_ID)
        x_train, y_train, x_test, y_test = self._get_train_test_split(
            x_feature_matrix, y_gloc_labels, experiment_metadata, num_splits, kfold_ID
        )

        return x_train, x_test, y_train, y_test, features["All"]

    def _impute_missing_data(
            self,
            gloc_data_all_features_numpy: np.ndarray,
            gloc_labels_numpy: np.ndarray,
            experiment_metadata: Dict[str, Any],
            impute_file_name: str,
            save_impute: bool,
            load_impute: bool,
            num_splits: int,
            kfold_ID: int,
            n_neighbors: int,
    ) -> np.ndarray:
        # Load or compute imputed features
        impute_path = self._resolve_advanced_impute_path(impute_file_name, kfold_ID)

        # NOTE: impute_path is derived from impute_file_name and kfold_ID in Processed Data.
        if load_impute and os.path.exists(impute_path):
            with open(impute_path, 'rb') as f:
                gloc_data_all_features_imputed_numpy = pickle.load(f)
            logger.info("Loaded imputed data from %s.", impute_path)
        else:
            # Only compute train/test indices when actually imputing
            _, _, _, _, train_indices, test_indices = self._groupedtrial_kfold_split(gloc_data_all_features_numpy, gloc_labels_numpy, num_splits, kfold_ID, experiment_metadata)
            gloc_data_all_features_imputed_numpy = self._faster_knn_impute_train_test(gloc_data_all_features_numpy, train_indices, test_indices, n_neighbors)

            del gloc_data_all_features_numpy # Free memory of original data after imputation

            if save_impute:
                impute_dir = os.path.dirname(impute_path)
                if impute_dir:
                    os.makedirs(impute_dir, exist_ok = True)
                with open(impute_path, 'wb') as f:
                    pickle.dump(gloc_data_all_features_imputed_numpy, f)
                logger.info("Saved imputed data to %s.", impute_path)

        return gloc_data_all_features_imputed_numpy

    def _resolve_advanced_impute_path(self, impute_file_name: str, kfold_ID: int) -> str:
        """Build advanced cache path in data_path/Processed Data with prefix and k-fold suffix."""
        processed_dir = Path(self.data_path) / "Processed Data"
        base_name = Path(impute_file_name)
        if base_name.suffix:
            file_name = f"advanced_{base_name.stem}_kfold_{kfold_ID}{base_name.suffix}"
        else:
            file_name = f"advanced_{base_name.name}_kfold_{kfold_ID}.pkl"
        return str((processed_dir / file_name).resolve())

    def _groupedtrial_kfold_split(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            num_splits: int,
            kfold_ID: int,
            experiment_metadata: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test using stratified group K-fold on trial groups."""
        # Grouped K-Fold setup (shuffle=False for reproducibility)
        gkf = StratifiedGroupKFold(n_splits = num_splits, shuffle = False)

        # Validate kfold_ID
        n_folds = gkf.get_n_splits()
        if kfold_ID < 0 or kfold_ID >= n_folds:
            raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

        # Get train and test indices for the specified fold
        trials = experiment_metadata["trial_ints"].reshape(-1, 1)
        train_index, test_index = next(islice(gkf.split(X, Y, trials), kfold_ID, kfold_ID + 1))

        # Extract split data
        x_train, y_train = X[train_index], Y[train_index]
        x_test, y_test = X[test_index], Y[test_index]

        return x_train, x_test, y_train, y_test, train_index, test_index

    def _faster_knn_impute_train_test(
            self,
            X: np.ndarray,
            train_ind: np.ndarray,
            test_ind: np.ndarray,
            k: int = 5,
            M: int = 32,
            efSearch: int = 64,
    ) -> np.ndarray:
        """Impute missing values via FAISS KNN, training on train set only to prevent leakage."""
        # Split into train and test
        X_train = X[train_ind]
        X_test = X[test_ind]

        # Identify missing values
        mask_train = np.isnan(X_train)
        mask_test = np.isnan(X_test)

        # Temporary mean imputation for FAISS indexing
        mean_vals = np.nanmean(X_train, axis = 0)
        X_train_temp = np.where(mask_train, mean_vals, X_train)
        X_test_temp = np.where(mask_test, mean_vals, X_test)

        # Build FAISS HNSW index on training data
        d = X_train.shape[1]
        index = self._build_faiss_knn_index(d, M=M, efSearch=efSearch)
        
        index.add(X_train_temp.astype(np.float32))

        # Impute training data
        distances, indices = index.search(X_train_temp.astype(np.float32), k + 1)
        X_train_imputed = X_train.copy()
        for i in range(X_train.shape[0]):
            neighbors = indices[i, 1:]  # skip self
            for j in range(X_train.shape[1]):
                if mask_train[i, j]:
                    neighbor_values = X_train_temp[neighbors, j]
                    X_train_imputed[i, j] = np.nanmean(neighbor_values)

        # Impute test data
        distances_test, indices_test = index.search(X_test_temp.astype(np.float32), k)
        X_test_imputed = X_test.copy()
        for i in range(X_test.shape[0]):
            neighbors = indices_test[i]
            for j in range(X_test.shape[1]):
                if mask_test[i, j]:
                    neighbor_values = X_train_temp[neighbors, j]
                    X_test_imputed[i, j] = np.nanmean(neighbor_values)

        # Rebuild into single array
        X_imputed = X.copy()
        X_imputed[train_ind] = X_train_imputed
        X_imputed[test_ind] = X_test_imputed

        return X_imputed
        
    def _generate_features(
            self,
            baseline_methods_to_use: List[str],
            combined_baseline: Dict[str, np.ndarray],
            combined_baseline_names: List[str],
            experiment_metadata: Dict[str, Any],
            output_feature_dtype: np.dtype = np.dtype(np.float32),
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate feature matrices from baseline data using only unengineered data streams."""
        # Get trial_order from metadata (passed from baseline_data)
        trial_order = experiment_metadata.get("trial_order")
        trial_column = experiment_metadata["trial_id"]
        
        if trial_order is None:
            # Fallback: Use pandas unique to preserve first-appearance order like legacy code
            trial_order = pd.unique(trial_column)
        
        # Concatenate trial arrays along first axis
        x_feature_matrix = np.concatenate(
            [combined_baseline[tid] for tid in trial_order], 
            axis = 0
        ).astype(output_feature_dtype)
        
        # Build baseline suffixes as frozenset for faster membership testing
        baseline_suffixes = frozenset(baseline_methods_to_use)
        
        # Use boolean indexing instead of nested loops
        ue_indices = np.array([
            i for i, feature in enumerate(combined_baseline_names)
            if feature in self._UNENGINEERED_STREAMS or self._is_baselined_stream(
                feature, baseline_suffixes
            )
        ])
        
        # Compute trial_ints in the same order as features were concatenated (trial_order).
        trial_ids_for_rows = np.concatenate([
            np.full(combined_baseline[tid].shape[0], tid, dtype=object)
            for tid in trial_order
        ])
        trial_ints = self._convert_to_unique_ordered_integers(trial_ids_for_rows)
        
        x_feature_matrix = x_feature_matrix[:, ue_indices]
        x_feature_matrix = np.hstack([
            x_feature_matrix,
            trial_ints.reshape(-1, 1)
        ])
        
        all_features = [combined_baseline_names[i] for i in ue_indices]
        
        return x_feature_matrix, all_features
    
    def _is_baselined_stream(self, feature_name: str, baseline_suffixes: frozenset) -> bool:
        """Check if feature_name matches '{unengineered_stream}_{baseline_method}' pattern."""
        # Early exit if feature doesn't contain underscore (optimization)
        if '_' not in feature_name:
            return False
        
        # Extract potential stream and suffix
        parts = feature_name.rsplit('_', 1)
        if len(parts) != 2:
            return False
        
        stream_candidate, suffix = parts
        
        # Check if suffix is a baseline method and stream is unengineered
        return suffix in baseline_suffixes and stream_candidate in self._UNENGINEERED_STREAMS

    def _feature_clean_and_prep(
            self,
            x_feature_matrix: np.ndarray,
            gloc_labels_numpy: np.ndarray,
            features: Dict[str, List[str]],
            experiment_metadata: Dict[str, Any],
            model_type: ModelType,
            should_impute: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Remove constant columns, optionally add AFE indicator, and remove NaN rows."""
        # CRITICAL: Extract trial_ints BEFORE removing constant columns,
        # otherwise the last column might be removed and we'll extract the wrong column
        trial_ints = x_feature_matrix[:, -1].copy()
        
        # Separate trial_ints from feature matrix for constant column removal
        # The last column is trial_ints which was added by _generate_features
        x_features_without_trials = x_feature_matrix[:, :-1]
        feature_names_without_trials = features["All"]  # features["All"] should NOT include trial_ints
        
        # Remove constant columns (typically no constant columns)
        x_features_without_trials, feature_names_without_trials = self._remove_constant_columns(
            x_features_without_trials, feature_names_without_trials
        )

        # Add AFE_indicator as 2nd-to-last column (before trial_ints) for explicit only
        # Since trial_ints is already separated, just append AFE_indicator at the end of features
        if model_type.afe_filter == "Complete" and model_type.feature_set == "Explicit":
            x_features_without_trials = np.hstack([
                x_features_without_trials,
                experiment_metadata["AFE_indicator"].reshape(-1, 1),
            ])

        # Restore trial_ints as the last column
        x_feature_matrix = np.hstack([x_features_without_trials, trial_ints.reshape(-1, 1)])
        all_features = feature_names_without_trials  # Don't include trial_ints in feature names

        # List-wise deletion or clean any residual NaNs
        if should_impute:
            # Remove rows with NaN
            x_feature_matrix_noNaN, y_gloc_labels_noNaN, all_features, trials_noNaN = self._process_NaN(
                x_feature_matrix,
                gloc_labels_numpy,
                all_features,
                trial_ints
            )
        else:
            x_feature_matrix_noNaN = x_feature_matrix
            y_gloc_labels_noNaN = gloc_labels_numpy
            trials_noNaN = trial_ints

        return x_feature_matrix_noNaN, y_gloc_labels_noNaN, all_features, trials_noNaN

    def _process_NaN(
            self,
            x_feature_matrix: np.ndarray,
            y_gloc_labels: np.ndarray,
            all_features: List[str],
            trials: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Remove all-NaN columns and any rows containing NaN values."""
        nan_mask = np.isnan(x_feature_matrix)

        # Find & remove columns if they have all NaN values
        index_column_all_NaN = nan_mask.all(axis=0)
        if index_column_all_NaN.any():
            x_feature_matrix = x_feature_matrix[:, ~index_column_all_NaN]
            all_features = [f for f, keep in zip(all_features, ~index_column_all_NaN) if keep]
            nan_mask = nan_mask[:, ~index_column_all_NaN]

        # Find & Remove rows in label/trial arrays if they have NaN values
        row_has_nan = nan_mask.any(axis=1)
        if row_has_nan.any():
            keep_rows = ~row_has_nan
            x_feature_matrix = x_feature_matrix[keep_rows]
            y_gloc_labels = y_gloc_labels[keep_rows]
            trials = trials[keep_rows]

        return x_feature_matrix, y_gloc_labels, all_features, trials

    def _get_train_test_split(
            self,
            x_feature_matrix: np.ndarray,
            y_gloc_labels: np.ndarray,
            experiment_metadata: Dict[str, Any],
            num_splits: int,
            kfold_ID: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train/test, standardize features, and preserve trial indices in the last column."""
        # Perform stratified group k-fold split
        x_train, x_test, y_train, y_test, _, _ = self._groupedtrial_kfold_split(
            x_feature_matrix, y_gloc_labels, num_splits, kfold_ID, experiment_metadata)

        # Extract trial indices from last column (use direct slicing for memory efficiency)
        train_trials = x_train[:, -1:]
        test_trials = x_test[:, -1:]
        
        # Remove trial column from features
        x_train = x_train[:, :-1]
        x_test = x_test[:, :-1]

        # Standardize features based on training data distribution
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Reattach trial indices as final column
        x_train = np.hstack([x_train, train_trials])
        x_test = np.hstack([x_test, test_trials])

        return x_train, y_train, x_test, y_test
    


class TraditionalDataPipeline(BaseGLOCDataPipeline):
    """Legacy-compatible data pipeline for temporal/traditional GLOC modeling."""

    # From Nikki paper
    _CLASSIFIER_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
        "LogReg": {
            "baseline_window": 5,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 5,
        },
        "RF": {
            "baseline_window": 18.75,
            "window_size": 7.5,
            "stride": 0.25,
            "feature_reduction_type": "none",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "LDA": {
            "baseline_window": 46.25,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "SVM": {
            "baseline_window": 32.5,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "ridge",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "EGB": {
            "baseline_window": 46.25,
            "window_size": 12.5,
            "stride": 0.25,
            "feature_reduction_type": "lasso",
            "baseline_methods_to_use": ["v0", "v1", "v2", "v5", "v6", "v7", "v8"],
            "imbalance_type": "none",
            "impute_type": 1,
            "n_neighbors": 3,
        },
        "KNN": {
            "baseline_window": 32.5,
            "window_size": 15,
            "stride": 0.25,
            "feature_reduction_type": "performance",
            "baseline_methods_to_use": ["v0", "v1", "v2"],
            "imbalance_type": "ros",
            "impute_type": 1,
            "n_neighbors": 5,
        },
    }

    def get_data(
            self,
            backstep: int,
            data_rate: int,
            classifier_type: str,
            model_type: ModelType,
            select_features: List[str],
            remove_NaN_trials: bool,
            offset: float,
            time_start: float,
            subject_to_analyze: Optional[str],
            trial_to_analyze: Optional[str],
            analysis_type: int,
            impute_file_name: Optional[str] = None,
            should_impute: bool = True,
            output_feature_dtype: np.dtype = np.dtype(np.float32),
            save_impute: bool = False,
            load_impute: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Return data for a given set of parameters."""
        (
            baseline_window,
            window_size,
            stride,
            _feature_reduction_type,
            baseline_methods_to_use,
            _imbalance_type,
            impute_type,
            n_neighbors,
        ) = self._get_hyperparameters_by_classifier(classifier_type)
        feature_groups_to_analyze = self._get_feature_groups(model_type)

        ############################################# LOAD AND PROCESS DATA #############################################
        logger.info("Loading and processing data with parameters: classifier_type=%s, model_type=%s, select_features=%s, remove_NaN_trials=%s, offset=%.2f, time_start=%.2f, subject_to_analyze=%s, trial_to_analyze=%s, analysis_type=%d",)
        # "Grabs GLOC event and predictor data, depending on 'analysis_type' and 'feature_groups_to_analyze"
        file_paths = self._get_data_locations()

        # Load data and slot in GOR EEG features from xlsx files, then filter to specified analysis type and process features based on specified feature groups
        gloc_data = self._load_data(file_paths, output_feature_dtype)
        gloc_data = self._filter_data_by_analysis_type(analysis_type, gloc_data, subject_to_analyze, trial_to_analyze)
        gloc_data, features = self._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths, output_feature_dtype)
        
        # Create GLOC categorical vector
        gloc_labels = self._label_gloc_events(gloc_data)

        # if is_complete_explicit: # Doesn't do anything since EEG shared features are used for complete_Explicit
        #     # Impute raw (using mean) value of the missing channels for each AFE condition
        #     gloc_data = self._eeg_specific_imputation(gloc_data, features)

        if model_type.afe_filter == "noAFE":
            # Reduce dataset based on AFE/noAFE condition
            gloc_data, gloc_labels = self._afe_subset(gloc_data, gloc_labels)

        ############################################### DATA CLEAN AND Some Imputation ###############################################
        logger.info("Cleaning data and performing imputation with should_impute=%s, n_neighbors=%d", should_impute, n_neighbors)
        if remove_NaN_trials:
            gloc_data, gloc_labels, _ = self._remove_all_nan_trials(gloc_data, features, gloc_labels)

        if should_impute:
            if impute_file_name is not None:
                traditional_impute_path = self._resolve_traditional_impute_path(impute_file_name, classifier_type)
            else:
                traditional_impute_path = None

            if load_impute and traditional_impute_path and os.path.exists(traditional_impute_path):
                with open(traditional_impute_path, 'rb') as f:
                    imputed_features = pickle.load(f)
                logger.info("Loaded traditional imputed data from %s.", traditional_impute_path)
            else:
                imputed_features = self._faster_knn_impute(
                    gloc_data[features["All"]].to_numpy(dtype=output_feature_dtype),
                    k=n_neighbors,
                )
                if save_impute and traditional_impute_path:
                    impute_dir = os.path.dirname(traditional_impute_path)
                    if impute_dir:
                        os.makedirs(impute_dir, exist_ok=True)
                    with open(traditional_impute_path, 'wb') as f:
                        pickle.dump(imputed_features, f)
                    logger.info("Saved traditional imputed data to %s.", traditional_impute_path)

            gloc_data[features["All"]] = imputed_features
        
        ################################################## REDUCE MEMORY ##################################################
        logger.info("Reducing memory usage by converting to numpy arrays with dtype=%s.", output_feature_dtype)
        # Extract out columns from gloc_data into experiment_metadata
        gloc_data_all_features_numpy, gloc_labels_numpy, experiment_metadata = self._reduce_memory(gloc_data, gloc_labels, features, output_feature_dtype)

        ###################################################### Prediction Offset ###############################################
        logger.info("Applying prediction offset with backstep=%d, data_rate=%d", backstep, data_rate)
        gloc_labels_numpy = self._y_prediction_offset(gloc_labels_numpy, backstep, data_rate, experiment_metadata["trial_id"])

        ################################################ BASELINE ################################################
        logger.info("Calculating baselines with methods: %s", baseline_methods_to_use)
        combined_baseline, combined_baseline_names, baseline_v0, baseline_names_v0 = self._get_combined_baseline_data(
            gloc_data_all_features_numpy,
            experiment_metadata,
            baseline_window,
            baseline_methods_to_use,
            features,
            file_paths,
            model_type
        )

        ################################# FEATURE GENERATION ########################################
        logger.info("Generating features with window_size=%.2f, stride=%.2f, offset=%.2f, time_start=%.2f", window_size, stride, offset, time_start)
        # Feature generation must run for each offset to window GLOC labels
        raw_gloc_labels_numpy = gloc_labels_numpy.copy()
        gloc_labels_numpy, gloc_data_all_features_numpy, features["All"] = self._feature_generation(
            time_start,
            offset,
            stride,
            window_size,
            combined_baseline,
            gloc_labels_numpy,
            experiment_metadata["trial_id"],
            experiment_metadata["Time (s)"],
            combined_baseline_names,
            baseline_names_v0,
            baseline_v0,
            feature_groups_to_analyze,
            output_feature_dtype
        )

        ################################################ Feature Reduction ################################################
        logger.info("Performing feature reduction with type: %s", _feature_reduction_type)
        # Add windowed AFE indicator if required by model type
        if model_type.afe_filter == "Complete" and model_type.feature_set == "Explicit":
            experiment_metadata["AFE_indicator_windowed"], _, _ = self._sliding_window_max(
                experiment_metadata["AFE_indicator"],
                experiment_metadata["trial_id"],
                experiment_metadata["Time (s)"],
                raw_gloc_labels_numpy,
                offset,
                stride,
                window_size,
                time_start
            )

            gloc_data_all_features_numpy = np.hstack([gloc_data_all_features_numpy, experiment_metadata["AFE_indicator_windowed"]])
            features["All"].append("AFE_indicator_windowed")

        # Backward compatibility: legacy feature lists may still reference "condition".
        translated_select_features = [
            feature_name.replace("condition", "AFE_indicator") for feature_name in select_features
        ]

        # Select columns by index to avoid an expensive full DataFrame materialization.
        feature_index = {feature_name: i for i, feature_name in enumerate(features["All"])}
        selected_indices = [feature_index[feature_name] for feature_name in translated_select_features]
        gloc_data_all_features_numpy = gloc_data_all_features_numpy[:, selected_indices]

        gloc_data_all_features_numpy, select_features = self._remove_constant_columns(gloc_data_all_features_numpy, translated_select_features)

        ################################################ NaN Processing ################################################
        logger.info("Processing NaN values temporally")
        gloc_labels_numpy, gloc_data_all_features_numpy, features["All"], _removed_ind = self._process_NaN_temporal(
            gloc_labels_numpy,
            gloc_data_all_features_numpy,
            select_features,
        )

        ################################################ Get Outputs Ready ############################################
        logger.info("Finalizing outputs and ensuring legacy compatibility in dtypes and shapes.")
        gloc_data_all_features_numpy, gloc_labels_numpy = self._ready_outputs(gloc_data_all_features_numpy, gloc_labels_numpy)

        return gloc_data_all_features_numpy, gloc_labels_numpy
    
    def _get_hyperparameters_by_classifier(
            self,
            classifier_type: str,
    ) -> Tuple[float, float, float, str, List[str], str, int, int]:
        """Return hyperparameters for a given classifier type."""
        params = self._CLASSIFIER_HYPERPARAMETERS.get(classifier_type)
        if params is None:
            available = ", ".join(sorted(self._CLASSIFIER_HYPERPARAMETERS.keys()))
            raise ValueError(f"Unknown classifier_type '{classifier_type}'. Available: {available}")

        return (
            params["baseline_window"],
            params["window_size"],
            params["stride"],
            params["feature_reduction_type"],
            params["baseline_methods_to_use"],
            params["imbalance_type"],
            params["impute_type"],
            params["n_neighbors"],
        )
    
    def _get_feature_groups(self, model_type: ModelType) -> Sequence[str]:
        feature_groups_to_analyze = self.FEATURE_GROUPS_BY_MODEL_TYPE[model_type]

        # NOTE:
        # AFE indicator is required for EEG imputation in complete models,
        # but is only included as a predictive feature for explicit models.

        return feature_groups_to_analyze

    def _resolve_traditional_impute_path(self, impute_file_name: str, classifier_type: str) -> str:
        """Build traditional cache path in data_path/Processed Data with prefix and model-name suffix."""
        processed_dir = Path(self.data_path) / "Processed Data"
        base_name = Path(impute_file_name)

        if base_name.suffix:
            file_name = f"traditional_{base_name.stem}_{classifier_type}{base_name.suffix}"
        else:
            file_name = f"traditional_{base_name.name}_{classifier_type}.pkl"

        return str((processed_dir / file_name).resolve())

    def _faster_knn_impute(
            self,
            X: np.ndarray,
            k: int = 5,
            M: int = 32,
            efSearch: int = 64,
    ) -> np.ndarray:
        """Impute missing values with FAISS KNN."""
        mask = np.isnan(X)
        X_imputed = X.copy()

        # Temporarily mean impute missing values
        X_temp = np.where(mask, np.nanmean(X, axis=0), X)

        # Build FAISS index (HNSW)
        d = X.shape[1] # dimension
        index = self._build_faiss_knn_index(d, M=M, efSearch=efSearch)
        
        index.add(X_temp.astype(np.float32))

        # Find k nearest neighbors
        distances, indices = index.search(X_temp.astype(np.float32), k + 1)

        # Impute missing values (skip self, which is always the first neighbor)
        for i in range(X.shape[0]):
            missing_cols = np.flatnonzero(mask[i])
            if missing_cols.size == 0:
                continue
            neighbors = indices[i, 1:] # skip self
            for j in missing_cols:
                neighbor_values = X_temp[neighbors, j]
                X_imputed[i, j] = np.nanmean(neighbor_values)

        return X_imputed
    
    def _y_prediction_offset(
            self,
            y: np.ndarray,
            backstep: int,
            data_rate: int,
            trial_set: np.ndarray,
    ) -> np.ndarray:
        """Shift GLOC labels left by the configured prediction horizon."""
        y = np.asarray(y).copy()
        offset = int(backstep * data_rate) # the actual number of indices to offset.
        # if backstep is given as seconds and data rate as hz
        # the result would be something like 5 seconds back * 25hz so 125 indices shift

        # y is passed as every single subject and trial in one so we have to break out the indices.

        unique_trials = np.unique(trial_set) # finds the unique trials within the set. Gives an array of name of each unique

        if offset <= 0:
            return y

        for trial in unique_trials:
            trial_indices = np.nonzero(trial_set == trial)[0]
            current_y = y[trial_indices]

            if not np.any(current_y):
                continue

            if offset >= current_y.shape[0]:
                y[trial_indices] = np.zeros_like(current_y)
                continue

            y_shifted = current_y[offset:]
            y[trial_indices] = np.concatenate([y_shifted, np.zeros(offset, dtype=current_y.dtype)])[: current_y.shape[0]]

        return y

    def _feature_generation(
            self,
            time_start: float,
            offset: float,
            stride: float,
            window_size: float,
            combined_baseline: Dict[str, np.ndarray],
            gloc: np.ndarray,
            trial_column: np.ndarray,
            time_column: np.ndarray,
            combined_baseline_names: List[str],
            baseline_names_v0: Any,
            baseline_v0: Dict[str, np.ndarray],
            feature_groups_to_analyze: Sequence[str],
            output_feature_dtype: np.dtype = np.dtype(np.float32),
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate temporal engineered features from baseline data."""
        # Sliding Window Mean
        gloc_window, sliding_window_mean_s1, number_windows, all_features_mean_s1, sliding_window_mean_s2, all_features_mean_s2 = (
            self._sliding_window_mean_calc(time_start, offset, stride, window_size, combined_baseline, gloc, trial_column,
            time_column, combined_baseline_names))

        # Sliding Window Standard Deviation, Max, Range
        (sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1, all_features_stddev_s1,
        all_features_max_s1,
        all_features_range_s1, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2,
        all_features_stddev_s2, all_features_max_s2,
        all_features_range_s2) = (
            self._sliding_window_calc(time_start, stride, window_size, combined_baseline, trial_column, time_column,
                                      number_windows, combined_baseline_names))

        # Additional Features
        (all_features_additional_s1, sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
        sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
        sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
        sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
        sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1,
        sliding_window_cognitive_ies_s1,
        all_features_additional_s2, sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
        sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
        sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
        sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
        sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
        sliding_window_cognitive_ies_s2) = \
            (self._sliding_window_other_features(time_start, stride, window_size, trial_column, time_column,
                                            number_windows,
                                            baseline_names_v0, baseline_v0, feature_groups_to_analyze))

        # Unpack Dictionary into Array & combine features into one feature array
        y_gloc_labels, x_feature_matrix = self._unpack_dict(gloc_window, sliding_window_mean_s1, number_windows,
                                                            sliding_window_stddev_s1,
                                                            sliding_window_max_s1, sliding_window_range_s1,
                                                            sliding_window_integral_left_pupil_s1,
                                                            sliding_window_integral_right_pupil_s1,
                                                            sliding_window_consecutive_elements_mean_left_pupil_s1,
                                                            sliding_window_consecutive_elements_mean_right_pupil_s1,
                                                            sliding_window_consecutive_elements_max_left_pupil_s1,
                                                            sliding_window_consecutive_elements_max_right_pupil_s1,
                                                            sliding_window_consecutive_elements_sum_left_pupil_s1,
                                                            sliding_window_consecutive_elements_sum_right_pupil_s1,
                                                            sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1,
                                                            sliding_window_cognitive_ies_s1,
                                                            sliding_window_mean_s2, sliding_window_stddev_s2,
                                                            sliding_window_max_s2, sliding_window_range_s2,
                                                            sliding_window_integral_left_pupil_s2,
                                                            sliding_window_integral_right_pupil_s2,
                                                            sliding_window_consecutive_elements_mean_left_pupil_s2,
                                                            sliding_window_consecutive_elements_mean_right_pupil_s2,
                                                            sliding_window_consecutive_elements_max_left_pupil_s2,
                                                            sliding_window_consecutive_elements_max_right_pupil_s2,
                                                            sliding_window_consecutive_elements_sum_left_pupil_s2,
                                                            sliding_window_consecutive_elements_sum_right_pupil_s2,
                                                            sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
                                                            sliding_window_cognitive_ies_s2,
                                                            output_feature_dtype)

        # Combine all features into array
        all_features = (all_features_mean_s1 + all_features_stddev_s1 + all_features_max_s1 + all_features_range_s1 +
                        all_features_additional_s1 + all_features_mean_s2 + all_features_stddev_s2 + all_features_max_s2 +
                        all_features_range_s2 + all_features_additional_s2)

        return y_gloc_labels.astype(output_feature_dtype), x_feature_matrix.astype(output_feature_dtype), all_features

    def _inter_trial_standardization(
            self,
            feature_dictionary: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute inter-trial z-score standardization for each trial matrix."""

        # Find Unique Trial ID
        trial_id_in_data = list(feature_dictionary.keys())

        ## FIND INTER TRIAL MEAN AND STD. DEVIATION TO USE FOR INTER TRIAL STANDARDIZATION ##
        # To do this, I first unpack the combined_baseline dictionary &
        # Determine total length of new unpacked dictionary items
        total_rows = 0
        for i in range(np.size(trial_id_in_data)):
            total_rows += np.shape(feature_dictionary[trial_id_in_data[i]])[0]

        # Find number of columns (using non-empty dictionaries)
        num_cols = np.shape(feature_dictionary[trial_id_in_data[0]])[1]

        # Pre-allocate
        all_data = np.zeros((total_rows, num_cols))

        # Iterate through unique trial_id
        current_index = 0
        for i in range(np.size(trial_id_in_data)):

            # Find number of rows in trial
            num_rows = np.shape(feature_dictionary[trial_id_in_data[i]])[0]

            # Set rows and columns in x_feature_matrix equal to current dictionary
            all_data[current_index:num_rows + current_index,:] = feature_dictionary[trial_id_in_data[i]]

            # Increment row index
            current_index += num_rows

        # Find mean and stand deviation of all data
        inter_trial_mean = np.nanmean(all_data, axis = 0, keepdims=True)
        inter_trial_standard_deviation = np.nanstd(all_data, axis = 0, keepdims=True)

        # Build Dictionary for each trial_id
        sliding_window_s2 = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):
            # Get data from current trial key
            current_trial_data = feature_dictionary[trial_id_in_data[i]]

            # Find inter-trial z-score
            inter_trial_z_score = ((current_trial_data - inter_trial_mean)/inter_trial_standard_deviation)

            # Define dictionary item for trial_id
            sliding_window_s2[trial_id_in_data[i]] = inter_trial_z_score

        return sliding_window_s2

    def _sliding_window_mean_calc(
            self,
            time_start: float,
            offset: float,
            stride: float,
            window_size: float,
            combined_baseline: Dict[str, np.ndarray],
            gloc: np.ndarray,
            trial_column: np.ndarray,
            time_column: np.ndarray,
            combined_baseline_names: List[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.int32], List[str], Dict[str, np.ndarray], List[str]]:
        """Compute sliding-window mean features and aligned GLOC labels."""

        # Find Unique Trial ID
        trial_id_in_data = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        # Build Dictionary for each trial_id
        sliding_window_mean = dict()
        sliding_window_mean_s1 = dict()
        gloc_window = dict()
        number_windows = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):

            # Determine index from current trial_id
            current_index = (trial_column == trial_id_in_data[i])

            # Create time array based on current_index
            current_time = np.array(time_column)
            time_trimmed = current_time[current_index]

            # Find end time for specific trial
            time_end = np.max(time_trimmed)

            # Determine number of windows
            number_windows_current = np.int32(((time_end - offset) // stride) - (window_size // stride - 1))

            # Pre-allocate arrays
            sliding_window_mean_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
            gloc_window_current = np.zeros((number_windows_current, 1))

            # Create trimmed gloc data for the specific
            gloc_trimmed = gloc[(trial_column == trial_id_in_data[i])]

            # Define iteration time
            time_iteration = time_start

            # Iterate through all windows to compute relevant parameters
            for j in range(number_windows_current):

                # Find index for current window
                time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))
                current_combined_baseline = combined_baseline[trial_id_in_data[i]][time_period_feature]

                # Take nanmean for the window (one value per column (feature))
                sliding_window_mean_current[j,:] = np.nanmean(current_combined_baseline, axis = 0, keepdims=True)

                # Find the offset time for G-LOC label
                time_period_gloc = (((time_iteration + offset) <= time_trimmed) &
                                    (time_trimmed < (time_iteration + offset + window_size)))

                # Create engineered label set to 1 if any values in window are 1
                gloc_window_current[j] = np.any(gloc_trimmed[time_period_gloc])

                # Adjust iteration_time
                time_iteration = stride + time_iteration

            # Compute z-score to standardize (intra-trial standardization)
            # This was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
            # should be removed in separate code or during feature selection.
            sliding_window_mean_current_z_score = np.zeros(np.shape(sliding_window_mean_current))
            if np.any(np.nanstd(sliding_window_mean_current, axis = 0, keepdims=True) == 0):
                # Z-score columns that don't have zero standard deviation
                for col in range(np.shape(sliding_window_mean_current)[1]):
                    if np.nanstd(sliding_window_mean_current[:,col]) != 0:
                        sliding_window_mean_current_z_score[:,col] = ((sliding_window_mean_current[:,col] - np.nanmean(
                            sliding_window_mean_current[:,col])) / np.nanstd(sliding_window_mean_current[:,col]))
                    else:
                        sliding_window_mean_current_z_score[:, col] = np.zeros(np.shape(sliding_window_mean_current)[0])
            else:
                sliding_window_mean_current_z_score = ((sliding_window_mean_current - np.nanmean(sliding_window_mean_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_mean_current, axis = 0, keepdims=True))

            # Define dictionary item for trial_id
            sliding_window_mean_s1[trial_id_in_data[i]] = sliding_window_mean_current_z_score
            sliding_window_mean[trial_id_in_data[i]] = sliding_window_mean_current
            gloc_window[trial_id_in_data[i]] = gloc_window_current
            number_windows[trial_id_in_data[i]] = number_windows_current

            # Name all features (s1 (intra-trial) standardization)
            all_features_mean_s1 = [s + '_mean_s1' for s in combined_baseline_names]

        # Compute inter-trial standardization
        sliding_window_mean_s2 = self._inter_trial_standardization(sliding_window_mean)

        # Name all features (s1 (intra-trial) standardization)
        all_features_mean_s2 = [s + '_mean_s2' for s in combined_baseline_names]

        return gloc_window, sliding_window_mean_s1, number_windows, all_features_mean_s1, sliding_window_mean_s2, all_features_mean_s2

    def _sliding_window_calc(
            self,
            time_start: float,
            stride: float,
            window_size: float,
            combined_baseline: Dict[str, np.ndarray],
            trial_column: np.ndarray,
            time_column: np.ndarray,
            number_windows: Dict[str, np.int32],
            combined_baseline_names: List[str],
    ) -> Tuple[Any, ...]:
        """Compute sliding-window std/max/range features with s1 and s2 variants."""

        # Find Unique Trial ID
        trial_id_in_data = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        # Build Dictionary for each trial_id
        # Windowed data (no standardization)
        sliding_window_stddev = dict()
        sliding_window_max = dict()
        sliding_window_range = dict()

        # s1 = Intra Trial Standardization
        sliding_window_stddev_s1 = dict()
        sliding_window_max_s1 = dict()
        sliding_window_range_s1 = dict()

        # s2 = Intra Trial Standardization
        sliding_window_stddev_s2 = dict()
        sliding_window_max_s2 = dict()
        sliding_window_range_s2 = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):

            # Determine index from current trial_id
            current_index = (trial_column == trial_id_in_data[i])

            # Create time array based on current_index
            current_time = np.array(time_column)
            time_trimmed = current_time[current_index]

            # Determine number of windows
            number_windows_current = number_windows[trial_id_in_data[i]]

            # Pre-allocate arrays
            sliding_window_stddev_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
            sliding_window_max_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))
            sliding_window_range_current = np.zeros((number_windows_current, np.shape(combined_baseline[trial_id_in_data[i]])[1]))

            # Define iteration time
            time_iteration = time_start

            # Iterate through all windows to compute relevant parameters
            for j in range(number_windows_current):

                # Find index for current window
                time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

                # Find feature for current window
                current_combined_baseline = combined_baseline[trial_id_in_data[i]][time_period_feature]

                # Take nan stddev for the window (one value per column (feature))
                sliding_window_stddev_current[j,:] = np.nanstd(current_combined_baseline, axis = 0, keepdims=True)

                # Take nan max for the window (one value per column (feature))
                sliding_window_max_current[j, :] = np.nanmax(current_combined_baseline, axis=0, keepdims=True)

                # Take nan range for the window (one value per column (feature))
                sliding_window_range_current[j, :] = np.nanmax(current_combined_baseline, axis=0, keepdims=True) - np.nanmin(current_combined_baseline, axis=0, keepdims=True)

                # Adjust iteration_time
                time_iteration = stride + time_iteration

            # Compute z-score to standardize
            # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
            # should be removed in separate code or during feature selection
            # Standard Deviation
            sliding_window_stddev_current_z_score_s1 = np.zeros(np.shape(sliding_window_stddev_current))
            if np.any(np.nanstd(sliding_window_stddev_current, axis=0, keepdims=True) == 0):
                # Z-score columns that don't have zero standard deviation
                for col in range(np.shape(sliding_window_stddev_current)[1]):
                    if np.nanstd(sliding_window_stddev_current[:, col]) != 0:
                        sliding_window_stddev_current_z_score_s1[:, col] = ((sliding_window_stddev_current[:, col] - np.nanmean(sliding_window_stddev_current[:, col])) /
                                                                np.nanstd(sliding_window_stddev_current[:, col]))
                    else:
                        sliding_window_stddev_current_z_score_s1[:, col] = np.zeros(np.shape(sliding_window_stddev_current)[0])
            else:
                sliding_window_stddev_current_z_score_s1 = ((sliding_window_stddev_current - np.nanmean(sliding_window_stddev_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_stddev_current, axis = 0, keepdims=True))

            # Max
            sliding_window_max_current_z_score_s1 = np.zeros(np.shape(sliding_window_max_current))
            if np.any(np.nanstd(sliding_window_max_current, axis=0, keepdims=True) == 0):
                # Find columns with zero standard deviation
                for col in range(np.shape(sliding_window_max_current)[1]):
                    if np.nanstd(sliding_window_max_current[:, col]) != 0:
                        sliding_window_max_current_z_score_s1[:, col] = ((sliding_window_max_current[:, col] - np.nanmean(
                            sliding_window_max_current[:, col])) / np.nanstd(sliding_window_max_current[:, col]))
                    else:
                        sliding_window_max_current_z_score_s1[:, col] = np.zeros(np.shape(sliding_window_max_current)[0])
            else:
                sliding_window_max_current_z_score_s1 = ((sliding_window_max_current - np.nanmean(sliding_window_max_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_max_current, axis = 0, keepdims=True))
            # Range
            sliding_window_range_current_z_score_s1 = np.zeros(np.shape(sliding_window_range_current))
            if np.any(np.nanstd(sliding_window_range_current, axis=0, keepdims=True) == 0):
                # Find columns with zero standard deviation
                for col in range(np.shape(sliding_window_range_current)[1]):
                    if np.nanstd(sliding_window_range_current[:, col]) != 0:
                        sliding_window_range_current_z_score_s1[:, col] = ((sliding_window_range_current[:, col] - np.nanmean(
                            sliding_window_range_current[:, col])) / np.nanstd(sliding_window_range_current[:, col]))
                    else:
                        sliding_window_range_current_z_score_s1[:, col] = np.zeros(np.shape(sliding_window_range_current)[0])
            else:
                sliding_window_range_current_z_score_s1 = ((sliding_window_range_current - np.nanmean(sliding_window_range_current, axis = 0, keepdims=True))
                                                / np.nanstd(sliding_window_range_current, axis = 0, keepdims=True))


            # Define dictionary item for trial_id
            # No standardization
            sliding_window_stddev[trial_id_in_data[i]] = sliding_window_stddev_current
            sliding_window_max[trial_id_in_data[i]] = sliding_window_max_current
            sliding_window_range[trial_id_in_data[i]] = sliding_window_range_current

            # Intra-trial standardization
            sliding_window_stddev_s1[trial_id_in_data[i]] = sliding_window_stddev_current_z_score_s1
            sliding_window_max_s1[trial_id_in_data[i]] = sliding_window_max_current_z_score_s1
            sliding_window_range_s1[trial_id_in_data[i]] = sliding_window_range_current_z_score_s1

            # Name features
            all_features_stddev_s1 = [s + '_stddev_s1' for s in combined_baseline_names]
            all_features_max_s1 = [s + '_max_s1' for s in combined_baseline_names]
            all_features_range_s1 = [s + '_range_s1' for s in combined_baseline_names]

        # Inter trial standardization
        sliding_window_stddev_s2 = self._inter_trial_standardization(sliding_window_stddev)
        sliding_window_max_s2 = self._inter_trial_standardization(sliding_window_max)
        sliding_window_range_s2 = self._inter_trial_standardization(sliding_window_range)

        all_features_stddev_s2 = [s + '_stddev_s2' for s in combined_baseline_names]
        all_features_max_s2 = [s + '_max_s2' for s in combined_baseline_names]
        all_features_range_s2 = [s + '_range_s2' for s in combined_baseline_names]

        return (sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1, all_features_stddev_s1, all_features_max_s1,
                all_features_range_s1, sliding_window_stddev_s2, sliding_window_max_s2, sliding_window_range_s2, all_features_stddev_s2,
                all_features_max_s2, all_features_range_s2)

    def _sliding_window_other_features(
            self,
            time_start: float,
            stride: float,
            window_size: float,
            trial_column: np.ndarray,
            time_column: np.ndarray,
            number_windows: Dict[str, np.int32],
            baseline_names_v0: Any,
            baseline_v0: Dict[str, np.ndarray],
            feature_groups_to_analyze: Sequence[str],
    ) -> Tuple[Any, ...]:
        """Compute additional temporal features (eye tracking, ECG, and cognitive)."""

        # Find Unique Trial ID
        trial_id_in_data = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        # Accept either a direct v0 name list or the full baseline-name dict.
        if isinstance(baseline_names_v0, dict):
            baseline_names_v0 = baseline_names_v0.get("v0", [])

        if 'eyetracking' in feature_groups_to_analyze:
            # Find indices of left and right pupil
            index_left_pupil = baseline_names_v0.index('Pupil diameter left [mm] - Tobii_v0')
            index_right_pupil = baseline_names_v0.index('Pupil diameter right [mm] - Tobii_v0')

            # Define eyetracking feature names
            eye_tracking_features = ['Left Pupil Integral (Non-Baseline)', 'Right Pupil Integral (Non-Baseline)',
                                    'Left Pupil Mean of Consecutive Difference (Non-Baseline)',
                                    'Right Pupil Mean of Consecutive Difference (Non-Baseline)',
                                    'Left Pupil Max of Consecutive Difference (Non-Baseline)',
                                    'Right Pupil Max of Consecutive Difference (Non-Baseline)',
                                    'Left Pupil Sum of Consecutive Difference (Non-Baseline)',
                                    'Right Pupil Sum of Consecutive Difference (Non-Baseline)']
        else:
            eye_tracking_features = []

        if 'ECG' in feature_groups_to_analyze:
            # Find indices of HR
            index_hr = baseline_names_v0.index('HR (bpm) - Equivital_v0')

            # Define ECG feature names
            ecg_features = ['HRV (SDNN)', 'HRV (RMSSD)']# , 'HRV (PNN50)']. Removed PNN50 due to interpolation
        else:
            ecg_features = []

        if 'cognitive' in feature_groups_to_analyze:
            # Find indices of Cognitive Response Time and Correct
            index_response_time = baseline_names_v0.index('RespTime - Cog_v0')
            index_correct = baseline_names_v0.index('Correct - Cog_v0')

            # Define ECG feature names
            cognitive_features = ['Cognitive IES']
        else:
            cognitive_features = []

        # Build Dictionary for each trial_id
        # No Standardization
        sliding_window_integral_left_pupil = dict()
        sliding_window_integral_right_pupil = dict()
        sliding_window_consecutive_elements_mean_left_pupil = dict()
        sliding_window_consecutive_elements_mean_right_pupil = dict()
        sliding_window_consecutive_elements_max_left_pupil = dict()
        sliding_window_consecutive_elements_max_right_pupil = dict()
        sliding_window_consecutive_elements_sum_left_pupil = dict()
        sliding_window_consecutive_elements_sum_right_pupil = dict()
        sliding_window_hrv_sdnn = dict()
        sliding_window_hrv_rmssd = dict()
        # sliding_window_hrv_pnn50 = dict()
        sliding_window_cognitive_ies = dict()

        # Intra-trial standardization (s1)
        sliding_window_integral_left_pupil_s1 = dict()
        sliding_window_integral_right_pupil_s1 = dict()
        sliding_window_consecutive_elements_mean_left_pupil_s1 = dict()
        sliding_window_consecutive_elements_mean_right_pupil_s1 = dict()
        sliding_window_consecutive_elements_max_left_pupil_s1 = dict()
        sliding_window_consecutive_elements_max_right_pupil_s1 = dict()
        sliding_window_consecutive_elements_sum_left_pupil_s1 = dict()
        sliding_window_consecutive_elements_sum_right_pupil_s1 = dict()
        sliding_window_hrv_sdnn_s1 = dict()
        sliding_window_hrv_rmssd_s1 = dict()
        # sliding_window_hrv_pnn50_s1 = dict()
        sliding_window_cognitive_ies_s1 = dict()

        # Inter-trial standardization (s2)
        sliding_window_integral_left_pupil_s2 = dict()
        sliding_window_integral_right_pupil_s2 = dict()
        sliding_window_consecutive_elements_mean_left_pupil_s2 = dict()
        sliding_window_consecutive_elements_mean_right_pupil_s2 = dict()
        sliding_window_consecutive_elements_max_left_pupil_s2 = dict()
        sliding_window_consecutive_elements_max_right_pupil_s2 = dict()
        sliding_window_consecutive_elements_sum_left_pupil_s2 = dict()
        sliding_window_consecutive_elements_sum_right_pupil_s2 = dict()
        sliding_window_hrv_sdnn_s2 = dict()
        sliding_window_hrv_rmssd_s2 = dict()
        # sliding_window_hrv_pnn50_s2 = dict()
        sliding_window_cognitive_ies_s2 = dict()

        # Iterate through all unique trial_id
        for i in range(np.size(trial_id_in_data)):

            # Determine index from current trial_id
            current_index = (trial_column == trial_id_in_data[i])

            # Create time array based on current_index
            current_time = np.array(time_column)
            time_trimmed = current_time[current_index]

            # Determine number of windows
            number_windows_current = number_windows[trial_id_in_data[i]]

            # Pre-allocate arrays
            if 'eyetracking' in feature_groups_to_analyze:
                sliding_window_integral_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_integral_right_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_mean_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_mean_right_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_max_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_max_right_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_sum_left_pupil_current = np.zeros((number_windows_current, 1))
                sliding_window_consecutive_elements_sum_right_pupil_current = np.zeros((number_windows_current, 1))
            if 'ECG' in feature_groups_to_analyze:
                sliding_window_hrv_sdnn_current = np.zeros((number_windows_current, 1))
                sliding_window_hrv_rmssd_current = np.zeros((number_windows_current, 1))
                # sliding_window_hrv_pnn50_current = np.zeros((number_windows_current, 1))
            if 'cognitive' in feature_groups_to_analyze:
                sliding_window_cognitive_ies_current = np.zeros((number_windows_current, 1))

            # Define iteration time
            time_iteration = time_start

            # Iterate through all windows to compute relevant parameters
            for j in range(number_windows_current):

                # Find index for current window
                time_period_feature = (time_iteration <= time_trimmed) & (time_trimmed < (time_iteration + window_size))

                # Find non-baseline feature for current window
                feature_window_no_baseline = baseline_v0[trial_id_in_data[i]][time_period_feature]

                if 'ECG' in feature_groups_to_analyze:
                    # Compute HRV
                    rr_interval = 60000 / feature_window_no_baseline[:,index_hr]
                    sliding_window_hrv_sdnn_current[j] = np.nanstd(rr_interval)

                    successive_difference = np.diff(rr_interval)
                    sliding_window_hrv_rmssd_current[j] = np.sqrt(np.nanmean(successive_difference**2))

                    # Compute PNN50
                    # count_50ms_diff_current = np.sum(np.abs(successive_difference) > 50 * 0.04) # 50 times (1/sampling freqeuncy)
                    # sliding_window_hrv_pnn50_current[j] = (count_50ms_diff_current / len(successive_difference)) * 100

                if 'cognitive' in feature_groups_to_analyze:
                    # Compute IES (Inverse Efficiency Score)
                    sliding_window_cognitive_ies_current[j] = np.nanmean(feature_window_no_baseline[:,index_response_time]) / (np.nanmean(feature_window_no_baseline[:,index_correct]))

                if 'eyetracking' in feature_groups_to_analyze:
                    # Compute non-baseline pupil features
                    left_pupil_no_baseline = feature_window_no_baseline[:, index_left_pupil]
                    right_pupil_no_baseline = feature_window_no_baseline[:, index_right_pupil]

                    # Integral (using Trapezoid rule)
                    sliding_window_integral_left_pupil_current[j] = (window_size / 2) * (left_pupil_no_baseline[-1] + left_pupil_no_baseline[0])
                    sliding_window_integral_right_pupil_current[j] = (window_size / 2) * (right_pupil_no_baseline[-1] + right_pupil_no_baseline[0])

                    # Compute average difference between consecutive elements
                    left_pupil_consecutive_difference = np.diff(left_pupil_no_baseline)
                    left_pupil_consecutive_difference_full = np.append(left_pupil_consecutive_difference, np.nan)

                    right_pupil_consecutive_difference = np.diff(right_pupil_no_baseline)
                    right_pupil_consecutive_difference_full = np.append(right_pupil_consecutive_difference, np.nan)

                    sliding_window_consecutive_elements_mean_left_pupil_current[j] = np.nanmean(left_pupil_consecutive_difference_full)
                    sliding_window_consecutive_elements_mean_right_pupil_current[j] = np.nanmean(right_pupil_consecutive_difference_full)

                    # Compute max difference between consecutive elements
                    sliding_window_consecutive_elements_max_left_pupil_current[j] = np.nanmax(left_pupil_consecutive_difference_full)
                    sliding_window_consecutive_elements_max_right_pupil_current[j] = np.nanmax(right_pupil_consecutive_difference_full)

                    # Compute sum of difference between consecutive elements
                    sliding_window_consecutive_elements_sum_left_pupil_current[j] = np.nansum(left_pupil_consecutive_difference_full)
                    sliding_window_consecutive_elements_sum_right_pupil_current[j] = np.nansum(right_pupil_consecutive_difference_full)

                # Adjust iteration_time
                time_iteration = stride + time_iteration

            # Compute Z-score
            if 'eyetracking' in feature_groups_to_analyze:
                # Compute z-score to standardize integral left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_integral_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_integral_left_pupil_current))
                if np.any(np.nanstd(sliding_window_integral_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_integral_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_integral_left_pupil_current[:, col]) != 0:
                            sliding_window_integral_left_pupil_current_z_score[:, col] = ((sliding_window_integral_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_integral_left_pupil_current[:, col])) / np.nanstd(sliding_window_integral_left_pupil_current[:, col]))
                        else:
                            sliding_window_integral_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_integral_left_pupil_current)[0])
                else:
                    sliding_window_integral_left_pupil_current_z_score = ((sliding_window_integral_left_pupil_current - np.nanmean(sliding_window_integral_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_integral_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize integral right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_integral_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_integral_right_pupil_current))
                if np.any(np.nanstd(sliding_window_integral_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_integral_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_integral_right_pupil_current[:, col]) != 0:
                            sliding_window_integral_right_pupil_current_z_score[:, col] = ((sliding_window_integral_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_integral_right_pupil_current[:, col])) / np.nanstd(sliding_window_integral_right_pupil_current[:, col]))
                        else:
                            sliding_window_integral_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_integral_right_pupil_current)[0])
                else:
                    sliding_window_integral_right_pupil_current_z_score = ((sliding_window_integral_right_pupil_current - np.nanmean(sliding_window_integral_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_integral_right_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize mean of difference of consecutive elements-left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_mean_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_mean_left_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_mean_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_mean_left_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_mean_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_mean_left_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_mean_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_mean_left_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_mean_left_pupil_current_z_score = ((sliding_window_consecutive_elements_mean_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_mean_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_mean_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize mean of difference of consecutive elements-right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_mean_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_mean_right_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_mean_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_mean_right_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_mean_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_mean_right_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_mean_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_mean_right_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_mean_right_pupil_current_z_score = ((sliding_window_consecutive_elements_mean_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_mean_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_mean_right_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize max of difference of consecutive elements-left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_max_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_max_left_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_max_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_max_left_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_max_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_max_left_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_max_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_max_left_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_max_left_pupil_current_z_score = ((sliding_window_consecutive_elements_max_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_max_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_max_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize max of difference of consecutive elements-right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature
                sliding_window_consecutive_elements_max_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_max_right_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_max_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_max_right_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_max_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_max_right_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_max_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_max_right_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_max_right_pupil_current_z_score = ((sliding_window_consecutive_elements_max_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_max_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_max_right_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize sum of difference of consecutive elements-left pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_sum_left_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_sum_left_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_sum_left_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_sum_left_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_sum_left_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_sum_left_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_sum_left_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_sum_left_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_sum_left_pupil_current_z_score = ((sliding_window_consecutive_elements_sum_left_pupil_current - np.nanmean(sliding_window_consecutive_elements_sum_left_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_sum_left_pupil_current, axis = 0, keepdims=True))

                # Compute z-score to standardize sum of difference of consecutive elements-right pupil
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_consecutive_elements_sum_right_pupil_current_z_score = np.zeros(np.shape(sliding_window_consecutive_elements_sum_right_pupil_current))
                if np.any(np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_consecutive_elements_sum_right_pupil_current)[1]):
                        if np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current[:, col]) != 0:
                            sliding_window_consecutive_elements_sum_right_pupil_current_z_score[:, col] = ((sliding_window_consecutive_elements_sum_right_pupil_current[:, col] - np.nanmean(
                                            sliding_window_consecutive_elements_sum_right_pupil_current[:, col])) / np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current[:, col]))
                        else:
                            sliding_window_consecutive_elements_sum_right_pupil_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_consecutive_elements_sum_right_pupil_current)[0])
                else:
                    sliding_window_consecutive_elements_sum_right_pupil_current_z_score = ((sliding_window_consecutive_elements_sum_right_pupil_current - np.nanmean(sliding_window_consecutive_elements_sum_right_pupil_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_consecutive_elements_sum_right_pupil_current, axis = 0, keepdims=True))
            if 'ECG' in feature_groups_to_analyze:
                # Compute z-score to standardize hrv sdnn
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_hrv_sdnn_current_z_score = np.zeros(np.shape(sliding_window_hrv_sdnn_current))
                if np.any(np.nanstd(sliding_window_hrv_sdnn_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_hrv_sdnn_current)[1]):
                        if np.nanstd(sliding_window_hrv_sdnn_current[:, col]) != 0:
                            sliding_window_hrv_sdnn_current_z_score[:, col] = ((sliding_window_hrv_sdnn_current[:, col] - np.nanmean(
                                            sliding_window_hrv_sdnn_current[:, col])) / np.nanstd(sliding_window_hrv_sdnn_current[:, col]))
                        else:
                            sliding_window_hrv_sdnn_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_hrv_sdnn_current)[0])
                else:
                    sliding_window_hrv_sdnn_current_z_score = ((sliding_window_hrv_sdnn_current - np.nanmean(sliding_window_hrv_sdnn_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_hrv_sdnn_current, axis = 0, keepdims=True))

                # Compute z-score to standardize hrv rmssd
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_hrv_rmssd_current_z_score = np.zeros(np.shape(sliding_window_hrv_rmssd_current))
                if np.any(np.nanstd(sliding_window_hrv_rmssd_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_hrv_rmssd_current)[1]):
                        if np.nanstd(sliding_window_hrv_rmssd_current[:, col]) != 0:
                            sliding_window_hrv_rmssd_current_z_score[:, col] = ((sliding_window_hrv_rmssd_current[:, col] - np.nanmean(
                                            sliding_window_hrv_rmssd_current[:, col])) / np.nanstd(sliding_window_hrv_rmssd_current[:, col]))
                        else:
                            sliding_window_hrv_rmssd_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_hrv_rmssd_current)[0])
                else:
                    sliding_window_hrv_rmssd_current_z_score = ((sliding_window_hrv_rmssd_current - np.nanmean(sliding_window_hrv_rmssd_current, axis = 0, keepdims=True))
                                                    / np.nanstd(sliding_window_hrv_rmssd_current, axis = 0, keepdims=True))

                # # Compute z-score to standardize hrv pnn50
                # # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # # should be removed in separate code or during feature selection
                # sliding_window_hrv_pnn50_current_z_score = np.zeros(np.shape(sliding_window_hrv_pnn50_current))
                # if np.any(np.nanstd(sliding_window_hrv_pnn50_current, axis=0, keepdims=True) == 0):
                #     # Find columns with zero standard deviation
                #     for col in range(np.shape(sliding_window_hrv_pnn50_current)[1]):
                #         if np.nanstd(sliding_window_hrv_pnn50_current[:, col]) != 0:
                #             sliding_window_hrv_pnn50_current_z_score[:, col] = ((sliding_window_hrv_pnn50_current[:, col] - np.nanmean(
                #                             sliding_window_hrv_pnn50_current[:, col])) / np.nanstd(sliding_window_hrv_pnn50_current[:, col]))
                #         else:
                #             sliding_window_hrv_pnn50_current_z_score[:, col] = np.zeros(
                #                 np.shape(sliding_window_hrv_pnn50_current)[0])
                # else:
                #     sliding_window_hrv_pnn50_current_z_score = ((sliding_window_hrv_pnn50_current - np.nanmean(sliding_window_hrv_pnn50_current, axis=0, keepdims=True))
                #                                        / np.nanstd(sliding_window_hrv_pnn50_current, axis=0, keepdims=True))

            if 'cognitive' in feature_groups_to_analyze:
                # Compute z-score to standardize cognitive IES
                # If/else was implemented to prevent a divide by 0 NaN error from no standardization. Features in this category
                # should be removed in separate code or during feature selection
                sliding_window_cognitive_IES_current_z_score = np.zeros(np.shape(sliding_window_cognitive_ies_current))
                if np.any(np.nanstd(sliding_window_cognitive_ies_current, axis=0, keepdims=True) == 0):
                    # Find columns with zero standard deviation
                    for col in range(np.shape(sliding_window_cognitive_ies_current)[1]):
                        if np.nanstd(sliding_window_cognitive_ies_current[:, col]) != 0:
                            sliding_window_cognitive_IES_current_z_score[:, col] = ((sliding_window_cognitive_ies_current[:, col] - np.nanmean(
                                            sliding_window_cognitive_ies_current[:, col])) / np.nanstd(sliding_window_cognitive_ies_current[:, col]))
                        else:
                            sliding_window_cognitive_IES_current_z_score[:, col] = np.zeros(
                                np.shape(sliding_window_cognitive_ies_current)[0])
                else:
                    sliding_window_cognitive_IES_current_z_score = ((sliding_window_cognitive_ies_current - np.nanmean(sliding_window_cognitive_ies_current, axis=0, keepdims=True))
                                                    / np.nanstd(sliding_window_cognitive_ies_current, axis=0, keepdims=True))

            # Define dictionary item for trial_id
            if 'eyetracking' in feature_groups_to_analyze:
                # No standardization
                sliding_window_integral_left_pupil[trial_id_in_data[i]] = sliding_window_integral_left_pupil_current
                sliding_window_integral_right_pupil[trial_id_in_data[i]] = sliding_window_integral_right_pupil_current
                sliding_window_consecutive_elements_mean_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_left_pupil_current
                sliding_window_consecutive_elements_mean_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_right_pupil_current
                sliding_window_consecutive_elements_max_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_left_pupil_current
                sliding_window_consecutive_elements_max_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_right_pupil_current
                sliding_window_consecutive_elements_sum_left_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_left_pupil_current
                sliding_window_consecutive_elements_sum_right_pupil[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_right_pupil_current

                # Intra-Trial Standardization (s1)
                sliding_window_integral_left_pupil_s1[trial_id_in_data[i]] = sliding_window_integral_left_pupil_current_z_score
                sliding_window_integral_right_pupil_s1[trial_id_in_data[i]] = sliding_window_integral_right_pupil_current_z_score
                sliding_window_consecutive_elements_mean_left_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_left_pupil_current_z_score
                sliding_window_consecutive_elements_mean_right_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_mean_right_pupil_current_z_score
                sliding_window_consecutive_elements_max_left_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_left_pupil_current_z_score
                sliding_window_consecutive_elements_max_right_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_max_right_pupil_current_z_score
                sliding_window_consecutive_elements_sum_left_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_left_pupil_current_z_score
                sliding_window_consecutive_elements_sum_right_pupil_s1[trial_id_in_data[i]] = sliding_window_consecutive_elements_sum_right_pupil_current_z_score
            if 'ECG' in feature_groups_to_analyze:
                # No standardization
                sliding_window_hrv_sdnn[trial_id_in_data[i]] = sliding_window_hrv_sdnn_current
                sliding_window_hrv_rmssd[trial_id_in_data[i]] = sliding_window_hrv_rmssd_current
                # sliding_window_hrv_pnn50[trial_id_in_data[i]] = sliding_window_hrv_pnn50_current

                # Intra-Trial Standardization (s1)
                sliding_window_hrv_sdnn_s1[trial_id_in_data[i]] = sliding_window_hrv_sdnn_current_z_score
                sliding_window_hrv_rmssd_s1[trial_id_in_data[i]] = sliding_window_hrv_rmssd_current_z_score
                # sliding_window_hrv_pnn50_s1[trial_id_in_data[i]] = sliding_window_hrv_pnn50_current_z_score
            if 'cognitive' in feature_groups_to_analyze:
                # No standardization
                sliding_window_cognitive_ies[trial_id_in_data[i]] = sliding_window_cognitive_ies_current

                # Intra-Trial Standardization (s1)
                sliding_window_cognitive_ies_s1[trial_id_in_data[i]] = sliding_window_cognitive_IES_current_z_score

            # Name all features
            all_features_additional = eye_tracking_features + ecg_features + cognitive_features
            all_features_additional_s1 = [s + '_s1' for s in all_features_additional]

        # Inter-trial standardization (s2)
        if 'eyetracking' in feature_groups_to_analyze:
            sliding_window_integral_left_pupil_s2 = self._inter_trial_standardization(sliding_window_integral_left_pupil)
            sliding_window_integral_right_pupil_s2 = self._inter_trial_standardization(sliding_window_integral_right_pupil)
            sliding_window_consecutive_elements_mean_left_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_mean_left_pupil)
            sliding_window_consecutive_elements_mean_right_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_mean_right_pupil)
            sliding_window_consecutive_elements_max_left_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_max_left_pupil)
            sliding_window_consecutive_elements_max_right_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_max_right_pupil)
            sliding_window_consecutive_elements_sum_left_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_sum_left_pupil)
            sliding_window_consecutive_elements_sum_right_pupil_s2 = self._inter_trial_standardization(sliding_window_consecutive_elements_sum_right_pupil)
        if 'ECG' in feature_groups_to_analyze:
            sliding_window_hrv_sdnn_s2 = self._inter_trial_standardization(sliding_window_hrv_sdnn)
            sliding_window_hrv_rmssd_s2 = self._inter_trial_standardization(sliding_window_hrv_rmssd)
            # sliding_window_hrv_pnn50_s2 = self._inter_trial_standardization(sliding_window_hrv_pnn50)
        if 'cognitive' in feature_groups_to_analyze:
            sliding_window_cognitive_ies_s2 = self._inter_trial_standardization(sliding_window_cognitive_ies)

        all_features_additional_s2 = [s + '_s2' for s in all_features_additional]

        return (all_features_additional_s1, sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
        sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
        sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
        sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
        sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_cognitive_ies_s1,
        all_features_additional_s2, sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
        sliding_window_consecutive_elements_mean_left_pupil_s2, sliding_window_consecutive_elements_mean_right_pupil_s2,
        sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
        sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
        sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2,
        sliding_window_cognitive_ies_s2)

    def _unpack_dict(
            self,
            gloc_window: Dict[str, np.ndarray],
            sliding_window_mean_s1: Dict[str, np.ndarray],
            number_windows: Dict[str, np.int32],
            sliding_window_stddev_s1: Dict[str, np.ndarray],
            sliding_window_max_s1: Dict[str, np.ndarray],
            sliding_window_range_s1: Dict[str, np.ndarray],
            sliding_window_integral_left_pupil_s1: Dict[str, np.ndarray],
            sliding_window_integral_right_pupil_s1: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_mean_left_pupil_s1: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_mean_right_pupil_s1: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_max_left_pupil_s1: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_max_right_pupil_s1: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_sum_left_pupil_s1: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_sum_right_pupil_s1: Dict[str, np.ndarray],
            sliding_window_hrv_sdnn_s1: Dict[str, np.ndarray],
            sliding_window_hrv_rmssd_s1: Dict[str, np.ndarray],
            sliding_window_cognitive_ies_s1: Dict[str, np.ndarray],
            sliding_window_mean_s2: Dict[str, np.ndarray],
            sliding_window_stddev_s2: Dict[str, np.ndarray],
            sliding_window_max_s2: Dict[str, np.ndarray],
            sliding_window_range_s2: Dict[str, np.ndarray],
            sliding_window_integral_left_pupil_s2: Dict[str, np.ndarray],
            sliding_window_integral_right_pupil_s2: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_mean_left_pupil_s2: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_mean_right_pupil_s2: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_max_left_pupil_s2: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_max_right_pupil_s2: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_sum_left_pupil_s2: Dict[str, np.ndarray],
            sliding_window_consecutive_elements_sum_right_pupil_s2: Dict[str, np.ndarray],
            sliding_window_hrv_sdnn_s2: Dict[str, np.ndarray],
            sliding_window_hrv_rmssd_s2: Dict[str, np.ndarray],
            sliding_window_cognitive_ies_s2: Dict[str, np.ndarray],
            output_feature_dtype: np.dtype = np.dtype(np.float32),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack per-trial dictionaries into global label and feature matrices."""
        # Find Unique Trial ID
        trial_id_in_data = list(sliding_window_mean_s1.keys())

        # Determine total length of new unpacked dictionary items
        total_rows = 0
        for i in range(np.size(trial_id_in_data)):
            total_rows += number_windows[trial_id_in_data[i]]

        # Create tuple of all dictionaries
        all_feature_dictionaries = [sliding_window_mean_s1, sliding_window_stddev_s1, sliding_window_max_s1, sliding_window_range_s1,
                                    sliding_window_integral_left_pupil_s1, sliding_window_integral_right_pupil_s1,
                                    sliding_window_consecutive_elements_mean_left_pupil_s1, sliding_window_consecutive_elements_mean_right_pupil_s1,
                                    sliding_window_consecutive_elements_max_left_pupil_s1, sliding_window_consecutive_elements_max_right_pupil_s1,
                                    sliding_window_consecutive_elements_sum_left_pupil_s1, sliding_window_consecutive_elements_sum_right_pupil_s1,
                                    sliding_window_hrv_sdnn_s1, sliding_window_hrv_rmssd_s1, sliding_window_cognitive_ies_s1,
                                    sliding_window_mean_s2, sliding_window_stddev_s2, sliding_window_max_s2,sliding_window_range_s2,
                                    sliding_window_integral_left_pupil_s2, sliding_window_integral_right_pupil_s2,
                                    sliding_window_consecutive_elements_mean_left_pupil_s2,sliding_window_consecutive_elements_mean_right_pupil_s2,
                                    sliding_window_consecutive_elements_max_left_pupil_s2, sliding_window_consecutive_elements_max_right_pupil_s2,
                                    sliding_window_consecutive_elements_sum_left_pupil_s2, sliding_window_consecutive_elements_sum_right_pupil_s2,
                                    sliding_window_hrv_sdnn_s2, sliding_window_hrv_rmssd_s2, sliding_window_cognitive_ies_s2]

        # Find all non-empty dictionaries
        non_empty_feature_dictionaries = []
        for dictionary in all_feature_dictionaries:
            if dictionary:
                non_empty_feature_dictionaries.append(dictionary)

        # Find number of columns (using non-empty dictionaries)
        num_cols = 0
        for dictionary in range(len(non_empty_feature_dictionaries)):
            current_dictionary = non_empty_feature_dictionaries[dictionary]
            num_cols = num_cols + np.shape(current_dictionary[trial_id_in_data[0]])[1]

        # Pre-allocate
        x_feature_matrix = np.zeros((total_rows, num_cols), dtype=output_feature_dtype)
        y_gloc_labels = np.zeros((total_rows, 1), dtype=output_feature_dtype)

        # Iterate through unique trial_id
        current_index = 0
        for i in range(np.size(trial_id_in_data)):

            # Find number of rows in trial
            num_rows = np.shape(sliding_window_mean_s1[trial_id_in_data[i]])[0]

            # For all non-empty dictionaries, set specific rows equal to the dictionary item corresponding to trial_id
            column_index = 0
            for dictionary in range(len(non_empty_feature_dictionaries)):

                # Find current dictionary
                current_dictionary = non_empty_feature_dictionaries[dictionary]

                # Set rows and columns in x_feature_matrix equal to current dictionary
                x_feature_matrix[current_index:num_rows + current_index,
                column_index:np.shape(current_dictionary[trial_id_in_data[i]])[1] + column_index] = current_dictionary[trial_id_in_data[i]].astype(output_feature_dtype)

                # Increment column index
                column_index += np.shape(current_dictionary[trial_id_in_data[i]])[1]

            # Set corresponding gloc labels from current trial
            y_gloc_labels[current_index:num_rows+current_index, :] = gloc_window[trial_id_in_data[i]].astype(output_feature_dtype)

            # Increment row index
            current_index += num_rows

        return y_gloc_labels, x_feature_matrix

    def _reduce_features(
            self,
            model_type: ModelType,
            offset: float,
            stride: float,
            window_size: float,
            time_start: float,
            gloc_data_all_features_imputed_numpy: np.ndarray,
            gloc_labels: np.ndarray,
            features: Dict[str, List[str]],
            experiment_metadata: Dict[str, Any],
            select_features: List[str],
    ) -> np.ndarray:
        """Reduce feature matrix columns to the requested selected features."""
        if model_type.afe_filter == "Complete" and model_type.feature_set == "Explicit":
            afe_indicator_column_windowed, gloc_compare, _ = self._sliding_window_max(
                experiment_metadata["AFE_indicator"], experiment_metadata["trial_id"], experiment_metadata["Time (s)"], gloc_labels,
                offset, stride, window_size, time_start
            )
            gloc_data_all_features_imputed_numpy = np.hstack([gloc_data_all_features_imputed_numpy, afe_indicator_column_windowed])
            features["All"].append("AFE_indicator_windowed")

        # Convert feature matrix to DataFrame for column selection
        gloc_data_all_features_imputed_numpy = pd.DataFrame(gloc_data_all_features_imputed_numpy, columns = features["All"])
        gloc_data_all_features_imputed_numpy = gloc_data_all_features_imputed_numpy[select_features]
        gloc_data_all_features_imputed_numpy = gloc_data_all_features_imputed_numpy.to_numpy()

        return gloc_data_all_features_imputed_numpy

    def _sliding_window_max(
            self,
            data_array: np.ndarray,
            trial_column: np.ndarray,
            time_column: np.ndarray,
            label_array: np.ndarray,
            offset: float,
            stride: float,
            window_size: float,
            time_start: float = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute sliding-window max features and aligned labels."""

        trial_ids = pd.unique(trial_column)  # order-preserving, matching legacy script behavior

        all_features = []
        all_labels = []
        all_trials = []

        for trial_id in trial_ids:
            # Select rows for this trial
            trial_mask = (trial_column == trial_id)
            trial_times = np.array(time_column[trial_mask])
            trial_data = data_array[trial_mask, :]
            trial_gloc = np.array(label_array[trial_mask])  # replace with label column if different

            time_end = np.max(trial_times)
            number_windows = int(((time_end - offset) // stride) - (window_size // stride - 1))

            t = time_start
            for w in range(number_windows):
                # Feature window
                window_mask = (t <= trial_times) & (trial_times < t + window_size)
                window_features = np.nanmax(trial_data[window_mask, :], axis=0)

                # G-LOC window
                gloc_mask = ((t + offset) <= trial_times) & (trial_times < t + offset + window_size)
                window_label = np.any(trial_gloc[gloc_mask])

                all_features.append(window_features)
                all_labels.append(window_label)
                all_trials.append(trial_id)

                t += stride

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        all_trials = np.array(all_trials)

        return all_features, all_labels, all_trials
    
    def _process_NaN_temporal(
            self,
            y_gloc_labels: np.ndarray,
            x_feature_matrix: np.ndarray,
            all_features: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Drop all-NaN columns and rows containing any NaN values."""
        # Find & remove columns if they have all NaN values
        nan_test = np.isnan(x_feature_matrix)
        index_column_all_NaN = np.all(nan_test, axis=0)
        keep_columns = ~index_column_all_NaN
        x_feature_matrix_noNaN_cols = x_feature_matrix[:, keep_columns]

        # Adjust all_features to only include columns that don't have all NaN
        all_features = [feature_name for feature_name, keep in zip(all_features, keep_columns) if keep]

        # Identify rows with any NaNs
        row_nan_mask = np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)

        # Save indices of removed rows
        removed_row_indices = np.where(row_nan_mask)[0]

        # Keep only rows without NaNs
        x_feature_matrix_noNaN = x_feature_matrix_noNaN_cols[~row_nan_mask]
        y_gloc_labels_noNaN = y_gloc_labels[~row_nan_mask]

        return y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, removed_row_indices
    
    def _ready_outputs(
            self,
            x_feature_matrix: Any,
            y_gloc_labels: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize outputs to numpy arrays with expected shapes."""
        x_feature_matrix = (
            x_feature_matrix.to_numpy() if hasattr(x_feature_matrix, "to_numpy") else np.asarray(x_feature_matrix)
        )
        y_gloc_labels = (
            y_gloc_labels.to_numpy().ravel() if hasattr(y_gloc_labels, "to_numpy") else np.ravel(y_gloc_labels)
        )

        return x_feature_matrix, y_gloc_labels