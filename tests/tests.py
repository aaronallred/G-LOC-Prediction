import os
import warnings
import gc
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from itertools import islice
import faiss
import pickle
from scripts.GLOC_data_pipeline import load_and_prepare_data_advanced

from data_manager import DataManager

# Test Configuration Constants
NUM_SPLITS = 5
KFOLD_ID = 0
N_NEIGHBORS = 5
BASELINE_WINDOW = 32.5
BASELINE_METHODS = ["v0", "v1", "v2", "v5", "v6"]
USE_REDUCED_DATASET = False

# Feature Group Constants
IMPLICIT_FEATURE_GROUPS = {"ECG", "BR", "temp", "eyetracking", "rawEEG"}
EXPLICIT_FEATURE_GROUPS = IMPLICIT_FEATURE_GROUPS.union({"AFE", "G", "processedEEG", "demographics", "strain"})
COMPLETE_FEATURE_GROUPS = {"AFE"}

# Memory Management Utilities
@contextmanager
def memory_cleanup(*objects):
    """Context manager to ensure memory cleanup of large objects."""
    try:
        yield
    finally:
        for obj in objects:
            del obj
        gc.collect()

def clear_memory(*objects):
    """Explicitly clear memory for provided objects."""
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()

@pytest.fixture(scope="function", autouse=True)
def cleanup_memory():
    """Auto-cleanup fixture to trigger garbage collection after each test."""
    yield
    gc.collect()

def _pull_eeg_sets():
    # list of shared eeg channels
    raw_eeg_shared_features = [
        "Fz - EEG",
        "F3 - EEG",
        "C3 - EEG",
        "C4 - EEG",
        "CP1 - EEG",
        "CP2 - EEG",
        "T8 - EEG",
        "TP9 - EEG",
        "TP10 - EEG",
        "P7 - EEG",
        "P8 - EEG",
    ]

    processed_eeg_shared_features = [
        "Fz_delta - EEG",
        "Fz_theta - EEG",
        "Fz_alpha - EEG",
        "Fz_beta - EEG",
        "F3_delta - EEG",
        "F3_theta - EEG",
        "F3_alpha - EEG",
        "F3_beta - EEG",
        "C3_delta - EEG",
        "C3_theta - EEG",
        "C3_alpha - EEG",
        "C3_beta - EEG",
        "C4_delta - EEG",
        "C4_theta - EEG",
        "C4_alpha - EEG",
        "C4_beta - EEG",
        "CP1_delta - EEG",
        "CP1_theta - EEG",
        "CP1_alpha - EEG",
        "CP1_beta - EEG",
        "CP2_delta - EEG",
        "CP2_theta - EEG",
        "CP2_alpha - EEG",
        "CP2_beta - EEG",
        "T8_delta - EEG",
        "T8_theta - EEG",
        "T8_alpha - EEG",
        "T8_beta - EEG",
        "TP9_delta - EEG",
        "TP9_theta - EEG",
        "TP9_alpha - EEG",
        "TP9_beta - EEG",
        "TP10_delta - EEG",
        "TP10_theta - EEG",
        "TP10_alpha - EEG",
        "TP10_beta - EEG",
        "P7_delta - EEG",
        "P7_theta - EEG",
        "P7_alpha - EEG",
        "P7_beta - EEG",
        "P8_delta - EEG",
        "P8_theta - EEG",
        "P8_alpha - EEG",
        "P8_beta - EEG",
    ]

    # list of AFE only eeg channels
    raw_eeg_afe_only = ["F4 - EEG", "T7 - EEG", "O1 - EEG", "O2 - EEG"]

    processed_eeg_afe_only = [
        "F4_delta - EEG",
        "F4_theta - EEG",
        "F4_alpha - EEG",
        "F4_beta - EEG",
        "T7_delta - EEG",
        "T7_theta - EEG",
        "T7_alpha - EEG",
        "T7_beta - EEG",
        "O1_delta - EEG",
        "O1_theta - EEG",
        "O1_alpha - EEG",
        "O1_beta - EEG",
        "O2_delta - EEG",
        "O2_theta - EEG",
        "O2_alpha - EEG",
        "O2_beta - EEG",
    ]

    # list of Non-AFE only eeg channels
    raw_eeg_nonafe_only = [
        "F1 - EEG",
        "AFz - EEG",
        "AF4 - EEG",
        "FT9 - EEG",
        "FT10 - EEG",
        "FC5 - EEG",
        "FC3 - EEG",
        "FC1 - EEG",
        "FC2 - EEG",
        "FC4 - EEG",
        "FC6 - EEG",
        "C5 - EEG",
        "Cz - EEG",
        "CP5 - EEG",
        "CP6 - EEG",
        "P5 - EEG",
        "P3 - EEG",
        "P1 - EEG",
        "Pz - EEG",
        "P4 - EEG",
        "P6 - EEG",
    ]

    processed_eeg_nonafe_only = [
        "F1_delta - EEG",
        "F1_theta - EEG",
        "F1_alpha - EEG",
        "F1_beta - EEG",
        "AFz_delta - EEG",
        "AFz_theta - EEG",
        "AFz_alpha - EEG",
        "AFz_beta - EEG",
        "AF4_delta - EEG",
        "AF4_theta - EEG",
        "AF4_alpha - EEG",
        "AF4_beta - EEG",
        "FT9_delta - EEG",
        "FT9_theta - EEG",
        "FT9_alpha - EEG",
        "FT9_beta - EEG",
        "FT10_delta - EEG",
        "FT10_theta - EEG",
        "FT10_alpha - EEG",
        "FT10_beta - EEG",
        "FC5_delta - EEG",
        "FC5_theta - EEG",
        "FC5_alpha - EEG",
        "FC5_beta - EEG",
        "FC3_delta - EEG",
        "FC3_theta - EEG",
        "FC3_alpha - EEG",
        "FC3_beta - EEG",
        "FC1_delta - EEG",
        "FC1_theta - EEG",
        "FC1_alpha - EEG",
        "FC1_beta - EEG",
        "FC2_delta - EEG",
        "FC2_theta - EEG",
        "FC2_alpha - EEG",
        "FC2_beta - EEG",
        "FC4_delta - EEG",
        "FC4_theta - EEG",
        "FC4_alpha - EEG",
        "FC4_beta - EEG",
        "FC6_delta - EEG",
        "FC6_theta - EEG",
        "FC6_alpha - EEG",
        "FC6_beta - EEG",
        "C5_delta - EEG",
        "C5_theta - EEG",
        "C5_alpha - EEG",
        "C5_beta - EEG",
        "Cz_delta - EEG",
        "Cz_theta - EEG",
        "Cz_alpha - EEG",
        "Cz_beta - EEG",
        "CP5_delta - EEG",
        "CP5_theta - EEG",
        "CP5_alpha - EEG",
        "CP5_beta - EEG",
        "CP6_delta - EEG",
        "CP6_theta - EEG",
        "CP6_alpha - EEG",
        "CP6_beta - EEG",
        "P5_delta - EEG",
        "P5_theta - EEG",
        "P5_alpha - EEG",
        "P5_beta - EEG",
        "P3_delta - EEG",
        "P3_theta - EEG",
        "P3_alpha - EEG",
        "P3_beta - EEG",
        "P1_delta - EEG",
        "P1_theta - EEG",
        "P1_alpha - EEG",
        "P1_beta - EEG",
        "Pz_delta - EEG",
        "Pz_theta - EEG",
        "Pz_alpha - EEG",
        "Pz_beta - EEG",
        "P4_delta - EEG",
        "P4_theta - EEG",
        "P4_alpha - EEG",
        "P4_beta - EEG",
        "P6_delta - EEG",
        "P6_theta - EEG",
        "P6_alpha - EEG",
        "P6_beta - EEG",
    ]

    return (
        processed_eeg_shared_features,
        processed_eeg_afe_only,
        processed_eeg_nonafe_only,
        raw_eeg_shared_features,
        raw_eeg_afe_only,
        raw_eeg_nonafe_only,
    )

def _read_and_process_demographics(demographic_data_filename, gloc_data_reduced):
    # Import demographics spreadsheet
    demographics = pd.read_csv(demographic_data_filename)
    demographics = demographics.astype({col: "float32" for col in demographics.select_dtypes(include="float64").columns})
    demographics = demographics.copy()

    # Grab variables of interest
    participant_index = demographics["GLOC ID"]  # Corresponds to subject 1-13
    participant_gender = pd.Series(demographics["Gender code [1/0]"])  # 0 = Female, 1 = Male
    participant_age = pd.Series(demographics["Age [yr]"])
    participant_height = pd.Series(demographics["height [m]"])
    participant_weight = pd.Series(demographics["weight (kg)"])
    participant_BMI = pd.Series(demographics["BMI [kg/m^2]"])
    participant_blood_volume = pd.Series(demographics["Blood Volume [L]"])  # Based on Nadler's approximation
    participant_SBP_seated = pd.Series(demographics["Resting SBP (seat)"])  # Systolic Blood Pressure
    participant_SBP_stand = pd.Series(demographics["resting SBP (stand)"])
    participant_SBP_exercise = pd.Series(demographics["SBP after squat"])
    participant_DBP_seated = pd.Series(demographics["Resting DBP (seat)"])  # Diastolic Blood Pressure
    participant_DBP_stand = pd.Series(demographics["resting DBP (stand)"])
    participant_DBP_exercise = pd.Series(demographics["DBP after squat"])
    participant_MAP_seated = pd.Series(demographics["Resting MAP"])  # Mean Arterial Pressure
    participant_MAP_stand = pd.Series(demographics["Resting MAP (stand"])
    participant_MAP_exercise = pd.Series(demographics["Post-Squat MAP"])
    participant_HR_seated = pd.Series(demographics["resting HR [seated]"])
    participant_HR_stand = pd.Series(demographics["resting HR (stand)"])
    participant_HR_exercise = pd.Series(demographics["HR after squat"])
    participant_max_leg_strength = pd.Series(demographics["Max (N)"])  # Max Leg Strength
    participant_largest_leg_circumference = pd.Series(demographics["largest leg circ. [cm]"])
    participant_lower_leg_volume = pd.Series(demographics["lower leg volume [mL]"])
    participant_skinfolds_chest_avg = pd.Series(demographics["chest avg"])  # Skin Folds to Approximate Body Fat %
    participant_skinfolds_abd_avg = pd.Series(demographics["abd avg"])
    participant_skinfolds_thigh_avg = pd.Series(demographics["thigh avg"])
    participant_skinfolds_midax_avg = pd.Series(demographics["midax avg"])
    participant_skinfolds_subscap_avg = pd.Series(demographics["subscap avg"])
    participant_skinfolds_tri_avg = pd.Series(demographics["tri avg"])
    participant_skinfolds_supra_avg = pd.Series(demographics["supra avg"])
    participant_skinfolds_sum = pd.Series(demographics["sum"])
    participant_percent_fat = pd.Series(demographics["% fat"])
    participant_leg_length = pd.Series(demographics["leg avg"])
    participant_arm_length = pd.Series(demographics["arm avg"])
    participant_midline_neck_length = pd.Series(demographics["neck (MNL) avg"])
    participant_lateral_neck_length = pd.Series(demographics["neck (LNL) avg"])
    participant_torso_length_post = pd.Series(demographics["torso (post) avg "])
    participant_torso_length_ax = pd.Series(demographics["torso (ax) avg "])
    participant_head_to_heart = pd.Series(demographics["head to heart avg"])
    participant_head_girth = pd.Series(demographics["head avg"])
    participant_neck_girth = pd.Series(demographics["neck avg"])
    participant_chest_upper_girth = pd.Series(demographics["chest upper avg"])
    participant_chest_under_girth = pd.Series(demographics["chest under avg"])
    participant_waist_girth = pd.Series(demographics["waist avg"])
    participant_hip_girth = pd.Series(demographics["hip avg"])
    participant_thigh_girth = pd.Series(demographics["thigh avg"])
    participant_calf_girth = pd.Series(demographics["calf avg"])
    participant_biceps_girth_flex = pd.Series(demographics["bicep flex avg"])
    participant_biceps_girth_relax = pd.Series(demographics["bicep relax avg"])
    participant_neck_flexion = pd.Series(demographics["avg (N) flexion"])
    participant_neck_extension = pd.Series(demographics["avg (N) extens"])
    participant_neck_right_rotation = pd.Series(demographics["avg (N) Rt. Rot"])
    participant_neck_left_rotation = pd.Series(demographics["avg (N) left rot"])
    participant_neck_left_lat_flex = pd.Series(demographics["avg (N) left lat flex"])
    participant_neck_right_lat_flex = pd.Series(demographics["avg (N) rt lat flex"])
    participant_pred_vo2 = pd.Series(demographics["pred. Vo2"])  # Predicted VO2

    # Concatenate all demographics of interest
    all_demographics = pd.concat(
        [
            participant_gender,
            participant_age,
            participant_height,
            participant_weight,
            participant_BMI,
            participant_blood_volume,
            participant_SBP_seated,
            participant_SBP_stand,
            participant_SBP_exercise,
            participant_DBP_seated,
            participant_DBP_stand,
            participant_DBP_exercise,
            participant_MAP_seated,
            participant_MAP_stand,
            participant_MAP_exercise,
            participant_HR_seated,
            participant_HR_stand,
            participant_HR_exercise,
            participant_max_leg_strength,
            participant_largest_leg_circumference,
            participant_lower_leg_volume,
            participant_skinfolds_chest_avg,
            participant_skinfolds_abd_avg,
            participant_skinfolds_thigh_avg,
            participant_skinfolds_midax_avg,
            participant_skinfolds_subscap_avg,
            participant_skinfolds_tri_avg,
            participant_skinfolds_supra_avg,
            participant_skinfolds_sum,
            participant_percent_fat,
            participant_leg_length,
            participant_arm_length,
            participant_midline_neck_length,
            participant_lateral_neck_length,
            participant_torso_length_post,
            participant_torso_length_ax,
            participant_head_to_heart,
            participant_head_girth,
            participant_neck_girth,
            participant_chest_upper_girth,
            participant_chest_under_girth,
            participant_waist_girth,
            participant_hip_girth,
            participant_thigh_girth,
            participant_calf_girth,
            participant_biceps_girth_flex,
            participant_biceps_girth_relax,
            participant_neck_flexion,
            participant_neck_extension,
            participant_neck_right_rotation,
            participant_neck_left_rotation,
            participant_neck_left_lat_flex,
            participant_neck_right_lat_flex,
            participant_pred_vo2,
        ],
        axis=1,
    )

    participant_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]

    # Initialize variables
    participant_demographics = np.zeros((len(gloc_data_reduced), all_demographics.shape[1]))
    length_previous_participant_data = -1

    # Organize all participant data in array same length as gloc_data_reduced
    for i in range(len(participant_list)):
        participant_identifier = participant_list[i]

        # Find all rows corresponding to the participant of interest
        index_list = gloc_data_reduced[gloc_data_reduced["subject"] == participant_identifier].index

        # Determine the length of the participant specific data
        length_current_participant_data = len(index_list)

        # Find which participant corresponds to the index of interest
        demographics_index = participant_index == int(participant_identifier)

        # Grab the demographics data for the current participant
        current_participant_demographics = all_demographics[demographics_index]

        # Copy participant demographics data to the full table with zeros
        participant_demographics[length_previous_participant_data + 1 : length_previous_participant_data + length_current_participant_data + 1, :] = current_participant_demographics

        length_previous_participant_data = length_current_participant_data + length_previous_participant_data

    # Create demographics data frame
    demographics_names = [
        "participant_gender",
        "participant_age",
        "participant_height",
        "participant_weight",
        "participant_BMI",
        "participant_blood_volume",
        "participant_SBP_seated",
        "participant_SBP_stand",
        "participant_SBP_exercise",
        "participant_DBP_seated",
        "participant_DBP_stand",
        "participant_DBP_exercise",
        "participant_MAP_seated",
        "participant_MAP_stand",
        "participant_MAP_exercise",
        "participant_HR_seated",
        "participant_HR_stand",
        "participant_HR_exercise",
        "participant_max_leg_strength",
        "participant_largest_leg_circumference",
        "participant_lower_leg_volume",
        "participant_skinfolds_chest_avg",
        "participant_skinfolds_abd_avg",
        "participant_skinfolds_thigh_avg",
        "participant_skinfolds_midax_avg",
        "participant_skinfolds_subscap_avg",
        "participant_skinfolds_tri_avg",
        "participant_skinfolds_supra_avg",
        "participant_skinfolds_sum",
        "participant_percent_fat",
        "participant_leg_length",
        "participant_arm_length",
        "participant_midline_neck_length",
        "participant_lateral_neck_length",
        "participant_torso_length_post",
        "participant_torso_length_ax",
        "participant_head_to_heart",
        "participant_head_girth",
        "participant_neck_girth",
        "participant_chest_upper_girth",
        "participant_chest_under_girth",
        "participant_waist_girth",
        "participant_hip_girth",
        "participant_thigh_girth",
        "participant_calf_girth",
        "participant_biceps_girth_flex",
        "participant_biceps_girth_relax",
        "participant_neck_flexion",
        "participant_neck_extension",
        "participant_neck_right_rotation",
        "participant_neck_left_rotation",
        "participant_neck_left_lat_flex",
        "participant_neck_right_lat_flex",
        "participant_pred_vo2",
    ]
    demographics_concat = pd.DataFrame(participant_demographics, columns=demographics_names)

    # Append all demographic data to gloc data reduced
    gloc_data_reduced = pd.concat([gloc_data_reduced, demographics_concat], axis=1)
    return gloc_data_reduced, demographics_names

def _process_strain_data(gloc_data_reduced):
    # Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet
    gloc_trial = gloc_data_reduced["trial_id"]
    magnitude_g = gloc_data_reduced["magnitude - Centrifuge"].to_numpy()
    event = gloc_data_reduced["event"]

    ######## Trial 04-06 (GLOC_Effectiveness stain value of 6.1g) ########
    trial_individual_coding = "04-06"
    g_level_strain = 6.1

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 06-02 (GLOC_Effectiveness stain value of 8.1g) ########
    trial_individual_coding = "06-02"
    g_level_strain = 8.1

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 06-06 End GOR Label in Event Validated ########
    trial_individual_coding = "06-06"

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])

    # Add missing 'end GOR' label
    return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

    # Find the index of new end GOR label in full length csv
    end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

    gloc_data_reduced.loc[end_GOR_label_index, "event_validated"] = "end GOR"

    ######## Trial 07-06 (GLOC_Effectiveness stain value of 4.6g) ########
    trial_individual_coding = "07-06"
    g_level_strain = 4.6

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 08-02 (GLOC_Effectiveness stain value of 8.3g) ########
    trial_individual_coding = "08-02"
    g_level_strain = 8.3

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 08-05 (GLOC_Effectiveness stain value of 4.3g) ########
    trial_individual_coding = "08-05"
    g_level_strain = 4.3

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    # Add missing 'end GOR' label
    return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]

    # Find the index of new end GOR label in full length csv
    end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

    gloc_data_reduced.loc[end_GOR_label_index, "event_validated"] = "end GOR"

    ######## Trial 08-06 (GLOC_Effectiveness stain value of 8.2g) ########
    trial_individual_coding = "08-06"
    g_level_strain = 8.2

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 09-03 (GLOC_Effectiveness stain value of 8.7g) ########
    trial_individual_coding = "09-03"
    g_level_strain = 8.7

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 09-05 (GLOC_Effectiveness stain value of 4.8g) ########
    trial_individual_coding = "09-05"
    g_level_strain = 4.8

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 10-05 (GLOC_Effectiveness stain value of 3.8g) ########
    trial_individual_coding = "10-05"
    g_level_strain = 3.8

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 10-06 (GLOC_Effectiveness stain value of 5.5g) ########
    trial_individual_coding = "10-06"
    g_level_strain = 5.5

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]
    current_event = event[trial_index]

    # Remove original label in event column
    mislabel_mask_strain = current_event.str.contains("strain during GOR")
    mislabel_index_strain = current_event[mislabel_mask_strain].index
    strain_relabel_index = mislabel_index_strain
    gloc_data_reduced.loc[strain_relabel_index, "event"] = None

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 11-02 (GLOC_Effectiveness stain value of 5.5g) ########
    trial_individual_coding = "11-02"
    g_level_strain = 5.5

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 12-03 (GLOC_Effectiveness stain value of 3.7g) ########
    trial_individual_coding = "12-03"
    g_level_strain = 3.7

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 13-02 (GLOC_Effectiveness stain value of 7.3g) ########
    trial_individual_coding = "13-02"
    g_level_strain = 7.3

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    ######## Trial 13-04 (GLOC_Effectiveness stain value of 7.4g) ########
    trial_individual_coding = "13-04"
    g_level_strain = 7.4

    # Find indices associated with current trial
    trial_index = gloc_data_reduced["trial_id"] == trial_individual_coding
    magnitude_g_trial = magnitude_g[trial_index]

    # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
    gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
    magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
    closest_value_strain_index = np.nanargmin(magnitude_difference)

    # Find the index of new strain label in full length csv
    strain_label_index = trial_index.idxmax() + closest_value_strain_index

    gloc_data_reduced.loc[strain_label_index, "event"] = "strain during GOR"

    return gloc_data_reduced, gloc_trial

def _get_expected_features(gloc_data_reduced, feature_groups_to_analyze, demographic_data_filename, model_type):
    # Get feature columns
    if "ECG" in feature_groups_to_analyze:
        ecg_features = [
            "HR (bpm) - Equivital",
            "ECG Lead 1 - Equivital",
            "ECG Lead 2 - Equivital",
            "HR_instant - Equivital",
            "HR_average - Equivital",
            "HR_w_average - Equivital",
        ]
    else:
        ecg_features = []

    if "BR" in feature_groups_to_analyze:
        br_features = ["BR (rpm) - Equivital"]
    else:
        br_features = []

    if "temp" in feature_groups_to_analyze:
        temp_features = ["Skin Temperature - IR Thermometer (°C) - Equivital"]
    else:
        temp_features = []

    if "fnirs" in feature_groups_to_analyze:
        fnirs_features = ["HbO2 - fNIRS", "Hbd - fNIRS"]

        # Generate additional fnirs specific features
        ox_deox_ratio = gloc_data_reduced["HbO2 - fNIRS"] / gloc_data_reduced["Hbd - fNIRS"]
        gloc_data_reduced["HbO2 / Hbd"] = ox_deox_ratio

        # append fnirs_features
        fnirs_features.append("HbO2 / Hbd")

        # output warning message for fnirs
        warnings.warn(
            "Per information from Chris on 01/15/25, FNIRS data was impacted by eye tracking glasses and should not be used."
        )
    else:
        fnirs_features = []

    if "eyetracking" in feature_groups_to_analyze:
        eyetracking_features = [
            "Pupil position left X [HUCS mm] - Tobii",
            "Pupil position left Y [HUCS mm] - Tobii",
            "Pupil position left Z [HUCS mm] - Tobii",
            "Pupil position right X [HUCS mm] - Tobii",
            "Pupil position right Y [HUCS mm] - Tobii",
            "Pupil position right Z [HUCS mm] - Tobii",
            "Pupil diameter left [mm] - Tobii",
            "Pupil diameter right [mm] - Tobii",
        ]

        # Generate additional pupil specific features
        pupil_difference = (
            gloc_data_reduced["Pupil diameter left [mm] - Tobii"]
            - gloc_data_reduced["Pupil diameter right [mm] - Tobii"]
        )
        gloc_data_reduced["Pupil Difference [mm]"] = pupil_difference

        # append eyetracking_features
        eyetracking_features.append("Pupil Difference [mm]")

    else:
        eyetracking_features = []

    # Adjust columns of data frame for feature always
    gloc_data_reduced.replace({"AFE_indicator": "N"}, 0, inplace=True)
    gloc_data_reduced.replace({"AFE_indicator": "AFE"}, 1, inplace=True)
    if "AFE" in feature_groups_to_analyze:
        afe_features = ["AFE_indicator"]

    else:
        afe_features = []

    if "G" in feature_groups_to_analyze and "Explicit" in model_type:
        # Process magnitude Centrifuge column to include 1.2g instead of NaN
        gloc_data_reduced.fillna({"magnitude - Centrifuge": 1.2}, inplace=True)

        # Grab g feature column
        g_features = ["magnitude - Centrifuge"]
    elif "G" in feature_groups_to_analyze and "Implicit" in model_type:
        # output warning message for implicit vs. explicit models
        warnings.warn("G cannot be used as a feature in implicit models. Feature removed.")

        g_features = []
    else:
        g_features = []

    if "rawEEG" in feature_groups_to_analyze:
        (
            _,
            _,
            _,
            raw_eeg_shared_features,
            raw_eeg_afe_only,
            raw_eeg_nonafe_only,
        ) = _pull_eeg_sets()
    else:
        raw_eeg_shared_features = []
        raw_eeg_afe_only = []
        raw_eeg_nonafe_only = []

    if "processedEEG" in feature_groups_to_analyze:
        (
            processed_eeg_shared_features,
            processed_eeg_afe_only,
            processed_eeg_nonafe_only,
            _,
            _,
            _,
        ) = _pull_eeg_sets()
    else:
        processed_eeg_shared_features = []
        processed_eeg_afe_only = []
        processed_eeg_nonafe_only = []

    if "AFE" in model_type:
        raw_eeg_condition_specific = raw_eeg_afe_only
        processed_eeg_condition_specific = processed_eeg_afe_only
    elif "noAFE" in model_type:
        raw_eeg_condition_specific = raw_eeg_nonafe_only
        processed_eeg_condition_specific = processed_eeg_nonafe_only
    else: # "complete" in model_type -> Use only used shared features
        raw_eeg_condition_specific = []
        processed_eeg_condition_specific = []

    if "strain" in feature_groups_to_analyze and "Explicit" in model_type:
        # For strain data, add missing strain labels before feature creation
        if USE_REDUCED_DATASET: # Skip processing of strain data if using reduced dataset since none to fill in
            gloc_data_reduced, gloc_trial = gloc_data_reduced, gloc_data_reduced["trial_id"]
        else:
            gloc_data_reduced, gloc_trial = _process_strain_data(gloc_data_reduced)

        # Create binary array for strain data
        event = gloc_data_reduced["event"]
        event_validated = gloc_data_reduced["event_validated"]
        strain_event = np.zeros(gloc_data_reduced.shape[0])

        # Find labeled 'strain' and 'end GOR' markings in the event column
        strain_indices = np.argwhere(event == "strain during GOR")
        end_GOR_indices = np.argwhere(event_validated == "end GOR")

        # Determine which trial strain label and end GOR label occur
        trial_strain = gloc_trial[strain_indices[:, 0]]
        trial_end_GOR = gloc_trial[end_GOR_indices[:, 0]]

        # When strain and end GOR label occur on the same trial, set chunk from
        # start of strain to end of GOR to 1
        for i in range(trial_strain.shape[0]):
            if trial_end_GOR.str.contains(trial_strain.iloc[i]).any():
                trial_end_GOR_contains_strain = trial_end_GOR.str.contains(trial_strain.iloc[i])
                end_gor_trial_index = trial_end_GOR_contains_strain[trial_end_GOR_contains_strain].index.tolist()
                strain_event[strain_indices[i, 0] : end_gor_trial_index[0]] = 1

        gloc_data_reduced["Strain [0/1]"] = strain_event

        # append strain features
        strain_features = ["Strain [0/1]"]
    elif "strain" in feature_groups_to_analyze and "Implicit" in model_type:
        # output warning message for implicit vs. explicit models
        warnings.warn("Strain cannot be used as a feature in implicit models. Feature removed.")

        strain_features = []
    else:
        strain_features = []

    if "demographics" in feature_groups_to_analyze and "Explicit" in model_type:
        # Read Demographics Spreadsheet and Append to gloc_data_reduced
        gloc_data_reduced, demographics_names = _read_and_process_demographics(demographic_data_filename, gloc_data_reduced)
        demographics_features = demographics_names
    elif "demographics" in feature_groups_to_analyze and "Implicit" in model_type:
        # output warning message for implicit vs. explicit models
        warnings.warn("Demographics cannot be used as a feature in implicit models. Feature removed.")

        demographics_features = []
    else:
        demographics_features = []

    # Combine names of different feature categories for baseline methods
    all_features = (
        ecg_features
        + br_features
        + temp_features
        + fnirs_features
        + eyetracking_features
        + afe_features
        + g_features
        + raw_eeg_shared_features
        + raw_eeg_condition_specific
        + processed_eeg_shared_features
        + processed_eeg_condition_specific
        + strain_features
        + demographics_features
    )
    all_features_phys = (
        ecg_features
        + br_features
        + temp_features
        + fnirs_features
        + eyetracking_features
        + raw_eeg_shared_features
        + raw_eeg_condition_specific
        + processed_eeg_shared_features
        + processed_eeg_condition_specific
    )
    all_features_ecg = ecg_features
    all_features_eeg = processed_eeg_shared_features + processed_eeg_condition_specific

    # Create matrix of all features for data being analyzed
    features = gloc_data_reduced[all_features].to_numpy(dtype=np.float32)
    features_phys = gloc_data_reduced[all_features_phys].to_numpy(dtype=np.float32)
    features_ecg = gloc_data_reduced[all_features_ecg].to_numpy(dtype=np.float32)
    features_eeg = gloc_data_reduced[all_features_eeg].to_numpy(dtype=np.float32)

    return (
        gloc_data_reduced,
        features,
        features_phys,
        features_ecg,
        features_eeg,
        all_features,
        all_features_phys,
        all_features_ecg,
        all_features_eeg,
    )

def _afe_subset_helper(model_type, gloc_data, all_features, gloc_labels):
    """
    Remove trials where AFE condition doesn't match the model type.
    Also returns filtered data and labels based on AFE condition.
    """
    cond = 1 if model_type[0] == 'AFE' else 0

    # All features and subject trial info to be put into a reduced dataframe
    all_features_with_ids = all_features + ['AFE_indicator', 'subject', 'trial']
    reduced_data_frame = gloc_data[all_features_with_ids]

    rows_to_remove = []
    N = 0  # number of trials total
    M = 0  # number of trials with missing data streams
    
    for (subject, trial), group in reduced_data_frame.groupby(['subject', 'trial']):
        trial_data = reduced_data_frame[
            (reduced_data_frame['subject'] == subject) & 
            (reduced_data_frame['trial'] == trial)
        ]

        # Check if the chosen AFE condition is violated at all during the trial
        if trial_data['AFE_indicator'].any().any() != cond:
            rows_to_remove.append(trial_data.index)
            M += 1

        N += 1

    # Flatten list of indices and remove them from the DataFrame
    rows_to_remove = [item for sublist in rows_to_remove for item in sublist]

    # Get rid of rows in the DF and array
    gloc_data = gloc_data.drop(rows_to_remove)
    gloc_data = gloc_data.reset_index(drop=True)
    gloc_labels = np.delete(gloc_labels, rows_to_remove, axis=0)

    print(f"There are {N - M} trials that match the chosen AFE condition out of {N} trials.")
    return gloc_data, gloc_labels

def _eeg_condition_impute_helper(gloc_data_reduced, all_features_eeg, afe_indicator_column, verbose=True):
    """
    Ensures both AFE (1) and non-AFE (0) conditions have the same feature columns.
    Missing columns are imputed with mean values.
    """
    df = gloc_data_reduced.copy()
    afe_mask = afe_indicator_column == 1
    nonafe_mask = afe_indicator_column == 0

    # Pull columns that need to be imputed for each type
    _, processed_eeg_afe_only, processed_eeg_nonafe_only, \
    _, raw_eeg_afe_only, raw_eeg_nonafe_only = _pull_eeg_sets()
    
    afe_only_cols = processed_eeg_afe_only + raw_eeg_afe_only
    nonafe_only_cols = processed_eeg_nonafe_only + raw_eeg_nonafe_only

    # Impute AFE-only columns for non-AFE rows
    for col in afe_only_cols:
        if col in all_features_eeg:
            mean_val = df.loc[afe_mask, col].mean(skipna=True)
            n_missing = df.loc[nonafe_mask, col].isna().sum()
            df.loc[nonafe_mask, col] = df.loc[nonafe_mask, col].fillna(mean_val)
            if verbose:
                print(f"Imputed {n_missing} values in '{col}' for non-AFE rows")

    # Impute non-AFE-only columns for AFE rows
    for col in nonafe_only_cols:
        if col in all_features_eeg:
            mean_val = df.loc[nonafe_mask, col].mean(skipna=True)
            n_missing = df.loc[afe_mask, col].isna().sum()
            df.loc[afe_mask, col] = df.loc[afe_mask, col].fillna(mean_val)
            if verbose:
                print(f"Imputed {n_missing} values in '{col}' for AFE rows")

    return df

def _remove_all_nan_trials_helper(gloc_data_reduced, all_features, gloc, verbose=True):
    """
    Remove trials where there is at least one data stream that is all NaN.
    Returns cleaned data, labels, and NaN proportion table.
    """
    all_features_with_ids = all_features + ['subject', 'trial']
    reduced_data_frame = gloc_data_reduced[all_features_with_ids]

    rows_to_remove = []
    nan_proportion_table = []
    N = 0  # number of trials total
    M = 0  # number of trials with missing data streams
    
    for (subject, trial), group in reduced_data_frame.groupby(['subject', 'trial']):
        trial_data = reduced_data_frame[
            (reduced_data_frame['subject'] == subject) & 
            (reduced_data_frame['trial'] == trial)
        ]

        # Compute proportion of NaN values for each feature
        nan_proportions = trial_data[all_features].isna().mean().to_dict()
        nan_proportion_table.append({'subject-trial': f"{subject}-{trial}", **nan_proportions})

        # Check if any columns in trial data are entirely NaN
        all_nan_cols = trial_data[all_features].isna().all()
        if all_nan_cols.any():
            rows_to_remove.append(trial_data.index)
            if verbose:
                nan_features = all_nan_cols[all_nan_cols].index.tolist()
                print(f"Subject {subject}, Trial {trial}: features entirely NaN → {nan_features}")
            M += 1

        N += 1

    # Flatten list of indices and remove them
    rows_to_remove = [item for sublist in rows_to_remove for item in sublist]
    nan_proportion_df = pd.DataFrame(nan_proportion_table)

    # Remove rows from DataFrame and array
    gloc_data_reduced = gloc_data_reduced.drop(rows_to_remove).reset_index(drop=True)
    gloc = np.delete(gloc, rows_to_remove, axis=0)

    print(f"There are {M} trials with all NaNs for at least one feature out of {N} trials. {N - M} trials remaining.")
    return gloc_data_reduced, gloc, nan_proportion_df

def _convert_to_unique_ordered_integers(strings):
    """Convert string array to unique ordered integers."""
    mapping = {}
    result = []
    current_id = 1
    for s in strings:
        if s not in mapping:
            mapping[s] = current_id
            current_id += 1
        result.append(mapping[s])
    return np.array(result, dtype=np.float32)

def _groupedtrial_kfold_split(Y, X, trials, num_splits, kfold_ID):
    """
    Split X and Y matrices into training and test sets using Stratified Group K-Fold.
    """
    gkf = StratifiedGroupKFold(n_splits=num_splits, shuffle=False)
    
    # Safety check for kfold_ID
    n_folds = gkf.get_n_splits()
    if kfold_ID < 0 or kfold_ID >= n_folds:
        raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

    # Get train and test indices for specific fold
    train_index, test_index = next(islice(gkf.split(X, Y, trials), kfold_ID, kfold_ID + 1))
    
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]
    
    return x_train, x_test, y_train, y_test, train_index, test_index

def _faster_knn_impute_train_test(X, train_ind, test_ind, k=5, M=32, efSearch=64):
    """
    Impute missing values using FAISS KNN with training data only.
    
    Parameters:
    - X: input array
    - train_ind: training dataset indices
    - test_ind: test dataset indices  
    - k: number of neighbors
    - M, efSearch: HNSW graph parameters
    
    Returns:
    - X_imputed: imputed version of input array
    """
    X_train = X[train_ind]
    X_test = X[test_ind]

    # Masks for missing values
    mask_train = np.isnan(X_train)
    mask_test = np.isnan(X_test)

    # Temporary mean imputation for FAISS indexing
    mean_vals = np.nanmean(X_train, axis=0)
    X_train_temp = np.where(mask_train, mean_vals, X_train)
    X_test_temp = np.where(mask_test, mean_vals, X_test)

    # Use single thread for deterministic behavior
    faiss.omp_set_num_threads(1)

    # Build FAISS HNSW index on training data
    d = X_train.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efSearch = efSearch
    
    # Create and assign RNG with seed for reproducibility
    rng = faiss.RandomGenerator(42)
    index.hnsw.rng = rng
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

def _assert_feature_sets(features, expected_all_features, expected_all_features_phys, expected_all_features_ecg, expected_all_features_eeg):
    def _assert_set(actual, expected, label):
        actual_set = set(actual)
        expected_set = set(expected)
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        assert not missing and not extra, f"{label} mismatch: missing {missing}, extra {extra}"

    _assert_set(features["All"], expected_all_features, "All features")
    _assert_set(features["Phys"], expected_all_features_phys, "Phys features")
    _assert_set(features["ECG"], expected_all_features_ecg, "ECG features")
    _assert_set(features["EEG"], expected_all_features_eeg, "EEG features")



@pytest.fixture(scope="session")
def manager():
    """Create DataManager instance for testing."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(test_dir), "data")
    return DataManager(testing = True, data_path = data_path, use_reduced_dataset = USE_REDUCED_DATASET)

@pytest.fixture(scope="session")
def file_paths(manager):
    """Get file paths from DataManager."""
    return manager._get_data_locations()

@pytest.fixture(scope="session")
def test_dir():
    """Get test directory path."""
    return os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def gloc_data(manager, file_paths, test_dir):
    """Load or create gloc_data with caching."""
    data_pickle_path = os.path.join(test_dir, "testing_temp", "gloc_data.pkl")

    if os.path.exists(data_pickle_path):
        with open(data_pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded data from {data_pickle_path}")
        return data
    
    # Load and cache data
    data = manager._load_data(file_paths)
    os.makedirs(os.path.dirname(data_pickle_path), exist_ok=True)
    with open(data_pickle_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved data to {data_pickle_path}")
    
    return data

def _get_imputed_data_fixture(model_type, manager, file_paths, gloc_data, test_dir):
    """Helper function to create imputed data for a specific model type."""
    # Create model-specific cache path
    model_str = f"{model_type[0]}_{model_type[1]}"
    impute_path = os.path.join(test_dir, "testing_temp", f"gloc_data_imputed_{model_str}.pkl")
    
    # Try loading cached results
    if os.path.exists(impute_path):
        cache_files = {
            'data': impute_path,
            'features': os.path.join(test_dir, "testing_temp", f"features_{model_str}.pkl"),
            'labels': os.path.join(test_dir, "testing_temp", f"gloc_labels_{model_str}.pkl"),
            'metadata': os.path.join(test_dir, "testing_temp", f"experiment_metadata_{model_str}.pkl")
        }
        
        try:
            with open(cache_files['data'], 'rb') as f:
                imputed_data = pickle.load(f)
            with open(cache_files['features'], 'rb') as f:
                features = pickle.load(f)
            with open(cache_files['labels'], 'rb') as f:
                labels = pickle.load(f)
            with open(cache_files['metadata'], 'rb') as f:
                metadata = pickle.load(f)
            print(f"Loaded imputed data for {model_str} from {impute_path}")
            return (imputed_data, labels, features, metadata)
        except Exception as e:
            print(f"Error loading cached data for {model_str}: {e}, recomputing...")
    
    # Compute imputed data
    data_copy = gloc_data.copy()
    
    # Get appropriate feature groups
    if model_type[1] == "Explicit":
        feature_groups = EXPLICIT_FEATURE_GROUPS
    else:  # Implicit
        feature_groups = IMPLICIT_FEATURE_GROUPS.union({"AFE"}) if model_type[0] == "Complete" else IMPLICIT_FEATURE_GROUPS
    
    data_copy, features = manager._process_and_get_feature_names(
        data_copy, feature_groups, model_type, file_paths
    )
    labels = manager._label_gloc_events(data_copy)
    
    if model_type[0] != "Complete":
        data_copy, labels = manager._afe_subset(data_copy, labels)
    
    data_numpy, labels_numpy, metadata = manager._reduce_memory(data_copy, labels, features)
    
    # Clear intermediate data
    del data_copy, labels
    gc.collect()
    
    # Cache intermediate results
    os.makedirs(os.path.join(test_dir, "testing_temp"), exist_ok=True)
    cache_data = [
        (features, f"features_{model_str}.pkl"),
        (labels_numpy, f"gloc_labels_{model_str}.pkl"),
        (metadata, f"experiment_metadata_{model_str}.pkl")
    ]
    
    for data_obj, filename in cache_data:
        path = os.path.join(test_dir, "testing_temp", filename)
        with open(path, 'wb') as f:
            pickle.dump(data_obj, f)
        print(f"Saved {filename}")
    
    # Impute data
    imputed_data = manager._impute_missing_data(
        data_numpy, labels_numpy, metadata, impute_path, 
        save_impute=True, load_impute=True, 
        num_splits=NUM_SPLITS, kfold_ID=KFOLD_ID, n_neighbors=N_NEIGHBORS
    )
    
    return (imputed_data, labels_numpy, features, metadata)



# Base Test Class with Common Tests
class TestDataManagerBase:
    """Base class with shared test logic for all model types."""
    MODEL_TYPE = None  # Override in subclasses
    __test__ = False  # Prevent pytest from collecting this base class
    
    @pytest.fixture(scope="class")
    def gloc_data_imputed_tuple(self, manager, file_paths, gloc_data, test_dir):
        """Load or create imputed data with caching for this model type."""
        result = _get_imputed_data_fixture(self.MODEL_TYPE, manager, file_paths, gloc_data, test_dir)
        yield result
        # Memory cleanup after class
        del result
        gc.collect()
    
    def test_get_feature_groups(self):
        """Test that feature groups are correctly determined for model type."""
        model_type = self.MODEL_TYPE
        if model_type[1] == "Explicit":
            expected = EXPLICIT_FEATURE_GROUPS
        else:  # Implicit
            expected = IMPLICIT_FEATURE_GROUPS.union({"AFE"}) if model_type[0] == "Complete" else IMPLICIT_FEATURE_GROUPS
        
        feature_groups, _ = DataManager(testing=True)._get_feature_groups_and_baseline_methods(model_type)
        assert set(feature_groups) == expected

    def test_getting_data_locations(self, manager, file_paths):
        def expected_data_locations(datafolder):
            if USE_REDUCED_DATASET:
                filename = os.path.join(datafolder, "all_trials_25_hz_stacked_null_str_filled_reduced.csv")
            else:
                filename = os.path.join(datafolder, "all_trials_25_hz_stacked_null_str_filled.csv")
            baseline_data_filename = os.path.join(datafolder, "ParticipantBaseline.csv")
            demographic_data_filename = os.path.join(datafolder, "GLOC_Effectiveness_Final.csv")

            list_of_eeg_data_files = [
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_01_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_02_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_03_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_04_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_05_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC4_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_06_DC6_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC4_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_07_DC6_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_08_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_08_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC5_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_09_DC6_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC2_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC4_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_10_DC5_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_11_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_12_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_12_DC5_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC1_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC3_25Hz_EEG_power_wMAR.xlsx"),
                os.path.join(datafolder, "GLOC_GOR_EEG_data_participants_1-13/GLOC_13_DC6_25Hz_EEG_power_wMAR.xlsx"),
            ]

            list_of_baseline_eeg_processed_files = [
                os.path.join(datafolder, "GLOC_EEG_baseline_delta_noAFE1.csv"),
                os.path.join(datafolder, "GLOC_EEG_baseline_theta_noAFE1.csv"),
                os.path.join(datafolder, "GLOC_EEG_baseline_alpha_noAFE1.csv"),
                os.path.join(datafolder, "GLOC_EEG_baseline_beta_noAFE1.csv"),
            ]

            return (
                filename,
                baseline_data_filename,
                demographic_data_filename,
                list_of_eeg_data_files,
                list_of_baseline_eeg_processed_files,
            )

        # Check if all expected keys are in the file paths
        expected_keys = {"main", "baseline", "demographic", "eeg_list", "baseline_eeg_processed_list"}
        assert set(file_paths.keys()) == expected_keys

        # Check against the actual file paths
        (
            filename,
            baseline_data_filename,
            demographic_data_filename,
            list_of_eeg_data_files,
            list_of_baseline_eeg_processed_files,
        ) = expected_data_locations(manager.data_path)
        assert file_paths["main"] == filename
        assert file_paths["baseline"] == baseline_data_filename
        assert file_paths["demographic"] == demographic_data_filename
        assert file_paths["eeg_list"] == list_of_eeg_data_files
        assert file_paths["baseline_eeg_processed_list"] == list_of_baseline_eeg_processed_files

    def test_loading_data(self, file_paths, gloc_data):
        def process_EEG_GOR(list_of_eeg_data_files, df):
            """
            This function slots in the GOR EEG data for the nonAFE condition based on the list of xlsx files.
            The NaNs in the initial csv are replaced.
            """
            # Initialize EEG dictionaries
            eeg_dict_delta = dict()
            eeg_dict_theta = dict()
            eeg_dict_alpha = dict()
            eeg_dict_beta = dict()

            # Iterate through all EEG files
            for file in range(len(list_of_eeg_data_files)):

                # Define current file
                current_file = list_of_eeg_data_files[file]

                # Grab corresponding trial based on file name
                # corresponding_trial = current_file[47] + current_file[48] + '-0' + current_file[52]
                corresponding_trial = current_file[-31] + current_file[-30] + '-0' + current_file[-26]

                # Define data frame for delta, theta, alpha, and beta bands
                df_delta = pd.read_excel(current_file, sheet_name='delta')
                df_theta = pd.read_excel(current_file, sheet_name='theta')
                df_alpha = pd.read_excel(current_file, sheet_name='alpha')
                df_beta = pd.read_excel(current_file, sheet_name='beta')

                # Remove time column from all spreadsheets that were read in
                df_delta = df_delta.iloc[:, :-1]
                df_theta = df_theta.iloc[:, :-1]
                df_alpha = df_alpha.iloc[:, :-1]
                df_beta = df_beta.iloc[:, :-1]

                # Add each data frame to dictionary corresponding to the trial
                eeg_dict_delta[corresponding_trial] = df_delta
                eeg_dict_theta[corresponding_trial] = df_theta
                eeg_dict_alpha[corresponding_trial] = df_alpha
                eeg_dict_beta[corresponding_trial] = df_beta

            # For each key in the dictionary, look at gloc_data_reduced for that trial
            all_trial_dictionary = list(eeg_dict_delta.keys())
            for key in range(len(all_trial_dictionary)):

                # Find current trial's data in gloc_data
                current_key = all_trial_dictionary[key]
                current_trial_data = df[df['trial_id'] == current_key]

                # Find first instance of 'begin GOR' in event_validated column for current trial
                event_validated_current_trial = np.array(current_trial_data['event_validated'])
                indices = np.argwhere(event_validated_current_trial == "begin GOR")
                if indices.size == 0:
                    continue  # or log warning / raise with more context
                index_begin_GOR = indices[0]

                # Find end index of GOR EEG data
                index_end_GOR_eeg = index_begin_GOR + len(eeg_dict_delta[current_key])

                # Iterate through all columns & insert data from Excel file
                column_names = eeg_dict_delta[current_key].columns
                for col in range(len(column_names)):

                    # Get current column name
                    column_name = column_names[col]

                    # Modify column name
                    modified_name_delta = column_name + '_delta' + ' - EEG'
                    modified_name_theta = column_name + '_theta' + ' - EEG'
                    modified_name_alpha = column_name + '_alpha' + ' - EEG'
                    modified_name_beta = column_name + '_beta' + ' - EEG'

                    # For each dictionary column, insert GOR EEG data in current_trial_data
                    # current_trial_data[modified_name_delta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_delta[current_key][column_name]
                    # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_delta] = eeg_dict_delta[current_key][column_name].astype(np.float32)
                    current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_delta)] = eeg_dict_delta[current_key][column_name].astype(np.float32)

                    # current_trial_data[modified_name_theta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_theta[current_key][column_name]
                    # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_theta] = eeg_dict_theta[current_key][column_name].astype(np.float32)
                    current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_theta)] = eeg_dict_theta[current_key][column_name].astype(np.float32)

                    # current_trial_data[modified_name_alpha][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_alpha[current_key][column_name]
                    # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_alpha] = eeg_dict_alpha[current_key][column_name].astype(np.float32)
                    current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_alpha)] = eeg_dict_alpha[current_key][column_name].astype(np.float32)

                    # current_trial_data[modified_name_beta][index_begin_GOR[0]:index_end_GOR_eeg[0]] = eeg_dict_beta[current_key][column_name]
                    # current_trial_data.loc[index_begin_GOR[0]:index_end_GOR_eeg[0], modified_name_beta] = eeg_dict_beta[current_key][column_name].astype(np.float32)
                    current_trial_data.iloc[index_begin_GOR[0]:index_end_GOR_eeg[0], current_trial_data.columns.get_loc(modified_name_beta)] = eeg_dict_beta[current_key][column_name].astype(np.float32)

                # Replace previously empty processed EEG data with current_trial_data
                df[df['trial_id'] == current_key] = current_trial_data

            return df

        # pickle file name
        pickle_filename = file_paths["main"].replace("csv", "pkl")

        # Check if pickle exists, if not create it
        if not os.path.isfile(pickle_filename):
            # Load CSV
            expected_gloc_data = pd.read_csv(file_paths["main"])
            expected_gloc_data = expected_gloc_data.astype({col: 'float32' for col in expected_gloc_data.select_dtypes(include='float64').columns})
            expected_gloc_data = expected_gloc_data.copy()
        else:
            # Load Pickle file
            expected_gloc_data = pd.read_pickle(pickle_filename)

        # Slot in GOR EEG data from other files
        expected_gloc_data = process_EEG_GOR(file_paths["eeg_list"], expected_gloc_data)
        
        # Adjust AFE condition column always
        expected_gloc_data["condition"] = expected_gloc_data["condition"].map({"N": 0, "AFE": 1})
        expected_gloc_data = expected_gloc_data.rename(columns = {"condition": "AFE_indicator"})

        # Convert float64 to float32 to save memory, and copy to defragment the DataFrame
        expected_gloc_data = expected_gloc_data.astype({col: "float32" for col in expected_gloc_data.select_dtypes(include = "float64").columns}).copy()
        
        # Extracting expected_gloc_data and trial into separate columns
        trial_ids = expected_gloc_data["trial_id"].to_numpy().astype("str")
        trial_ids = np.array(np.char.split(trial_ids, "-").tolist())
        expected_gloc_data["subject"] = trial_ids[:, 0]
        expected_gloc_data["trial"] = trial_ids[:, 1]
        expected_gloc_data = expected_gloc_data.copy()

        # Check if loaded data matches expected data
        pd.testing.assert_frame_equal(gloc_data, expected_gloc_data)
        assert gloc_data.equals(expected_gloc_data)

    def test_getting_feature_names(self, manager, file_paths, gloc_data):
        """Test feature name extraction for the model type."""
        data_copy = gloc_data.copy()
        model_type = self.MODEL_TYPE
        
        # Get appropriate feature groups for model type
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        
        gloc_data_processed, features = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )
        
        # Generate expected features using the original data
        _, _, _, _, _, expected_all_features, expected_all_features_phys, \
        expected_all_features_ecg, expected_all_features_eeg = _get_expected_features(
            gloc_data.copy(),
            feature_groups,
            file_paths["demographic"], 
            model_type
        )
        
        _assert_feature_sets(
            features, expected_all_features, expected_all_features_phys,
            expected_all_features_ecg, expected_all_features_eeg
        )
        
        # Explicit cleanup
        del data_copy, gloc_data_processed
        gc.collect()

    def test_gloc_labeling(self, manager, file_paths, gloc_data):
        """Test GLOC event labeling."""
        data_copy = gloc_data.copy()
        model_type = self.MODEL_TYPE
        
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        data_copy, _ = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )

        event_validated = data_copy["event_validated"].to_numpy()
        trial_id = data_copy["trial_id"].to_numpy()

        # Find indices where 'GLOC' and 'return to consciousness' occur
        gloc_indices = np.argwhere(event_validated == "GLOC")
        rtc_indices = np.argwhere(event_validated == "return to consciousness")

        expected_gloc_labels = np.zeros(event_validated.shape, dtype=np.float32)
        for i in range(gloc_indices.shape[0]):
            if trial_id[gloc_indices[i]] == trial_id[rtc_indices[i]]:
                expected_gloc_labels[gloc_indices[i, 0]:rtc_indices[i, 0]] = 1

        gloc_labels = manager._label_gloc_events(data_copy)

        np.testing.assert_array_equal(gloc_labels, expected_gloc_labels)
        assert np.array_equal(gloc_labels, expected_gloc_labels), \
            "The GLOC labels do not match the expected labels based on event_validated and trial_id."
        
        del data_copy, event_validated, trial_id
        gc.collect()

    def test_afe_subset(self, manager, file_paths, gloc_data):
        """Test AFE subset filtering."""
        model_type = self.MODEL_TYPE
        # Only test for noAFE models
        if model_type[0] == "Complete":
            pytest.skip("Test only applicable for non-Complete models")
        
        data_copy = gloc_data.copy()
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        
        data_copy, features = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )
        labels = manager._label_gloc_events(data_copy)

        # Create expected output using helper
        expected_data, expected_labels = _afe_subset_helper(
            model_type, data_copy.copy(), features["All"], labels.copy()
        )

        # Get actual output
        actual_data, actual_labels = manager._afe_subset(data_copy, labels)

        assert actual_data.equals(expected_data), \
            "The gloc_data after afe_subset does not match the expected gloc_data."
        assert np.array_equal(actual_labels, expected_labels), \
            "The gloc_labels after afe_subset does not match the expected gloc_labels."
        
        del data_copy, labels, expected_data, expected_labels
        gc.collect()

    def test_eeg_specific_imputation(self, manager, file_paths, gloc_data):
        """Test EEG-specific imputation."""
        model_type = self.MODEL_TYPE
        # Only test for Complete models
        if model_type[0] != "Complete":
            pytest.skip("Test only applicable for Complete models")
        
        data_copy = gloc_data.copy()
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        
        data_copy, features = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )
        labels = manager._label_gloc_events(data_copy)

        # Setup expected result
        expected_data = data_copy.copy()
        condition_idx = features["All"].index('AFE_indicator')
        afe_indicator_column = expected_data.iloc[:, condition_idx]

        # Impute using helper
        expected_data = _eeg_condition_impute_helper(
            expected_data, features["EEG"], afe_indicator_column, verbose=False
        )

        # Get actual output
        manager._eeg_specific_imputation(data_copy, features)

        assert data_copy.equals(expected_data), \
            "The gloc_data after eeg_specific_imputation does not match the expected gloc_data."
        
        del data_copy, labels, expected_data
        gc.collect()

    def test_remove_all_NaN_trials(self, manager, file_paths, gloc_data):
        """Test removal of trials with all NaN features."""
        data_copy = gloc_data.copy()
        model_type = self.MODEL_TYPE
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        
        data_copy, features = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )
        labels = manager._label_gloc_events(data_copy)
        
        if model_type[0] != "Complete":
            data_copy, labels = manager._afe_subset(data_copy, labels)

        # Create expected output
        expected_data, expected_labels, _ = _remove_all_nan_trials_helper(
            data_copy.copy(), features["All"], labels.copy(), verbose=False
        )

        # Get actual output
        manager._remove_all_nan_trials(data_copy, features, labels)

        assert data_copy.equals(expected_data), \
            "The gloc_data after remove_all_nan_trials does not match the expected gloc_data."
        assert np.array_equal(labels, expected_labels), \
            "The gloc_labels after remove_all_nan_trials does not match the expected gloc_labels."
        
        del data_copy, labels, expected_data, expected_labels
        gc.collect()

    def test_reduce_memory(self, manager, file_paths, gloc_data):
        """Test memory reduction and data conversion to numpy."""
        data_copy = gloc_data.copy()
        model_type = self.MODEL_TYPE
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        
        data_copy, features = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )
        labels = manager._label_gloc_events(data_copy)
        
        if model_type[0] != "Complete":
            data_copy, labels = manager._afe_subset(data_copy, labels)

        # Extract expected columns before reduction
        trial_column = data_copy['trial_id']
        trial_ints = _convert_to_unique_ordered_integers(trial_column)
        time_column = data_copy['Time (s)']
        event_validated_column = data_copy['event_validated']
        subject_column = data_copy['subject']
        afe_indicator_column = data_copy["AFE_indicator"].to_numpy(dtype=np.float32).reshape(-1, 1)
        expected_features_numpy = data_copy[features["All"]].to_numpy()

        # Get actual output
        actual_features_numpy, labels_numpy, metadata = manager._reduce_memory(
            data_copy, labels, features
        )

        # Verify all metadata columns match
        assert np.array_equal(trial_column, metadata["trial_id"]), \
            "The trial_id column in the experiment metadata does not match."
        assert np.array_equal(trial_ints, metadata["trial_ints"]), \
            "The trial_ints column in the experiment metadata does not match."
        assert np.array_equal(time_column, metadata["Time (s)"]), \
            "The time column in the experiment metadata does not match."
        assert np.array_equal(event_validated_column, metadata["event_validated"]), \
            "The event_validated column in the experiment metadata does not match."
        assert np.array_equal(subject_column, metadata["subject"]), \
            "The subject column in the experiment metadata does not match."
        assert np.array_equal(afe_indicator_column, metadata["AFE_indicator"]), \
            "The AFE_indicator column in the experiment metadata does not match."
        assert np.array_equal(actual_features_numpy, expected_features_numpy, equal_nan=True), \
            "The gloc_data_all_features_numpy after reduce_memory does not match."
        assert np.array_equal(labels_numpy, labels), \
            "The gloc_labels_numpy after reduce_memory does not match."
        
        del data_copy, labels, expected_features_numpy
        gc.collect()

    def test_impute_missing_data(self, manager, file_paths, gloc_data):
        """Test missing data imputation."""
        data_copy = gloc_data.copy()
        model_type = self.MODEL_TYPE
        model_str = f"{model_type[0]}_{model_type[1]}"
        impute_path = f"test_imputation_{model_str}.pkl"
        
        feature_groups, _ = manager._get_feature_groups_and_baseline_methods(model_type)
        data_copy, features = manager._process_and_get_feature_names(
            data_copy, feature_groups, model_type, file_paths
        )
        labels = manager._label_gloc_events(data_copy)
        
        if model_type[0] != "Complete":
            data_copy, labels = manager._afe_subset(data_copy, labels)
        
        features_numpy, labels_numpy, metadata = manager._reduce_memory(
            data_copy, labels, features
        )

        # Create expected output
        expected_features = features_numpy.copy()
        
        # Get train/test split
        _, _, _, _, train_ind, test_ind = _groupedtrial_kfold_split(
            labels_numpy, expected_features, metadata["trial_ints"],
            NUM_SPLITS, KFOLD_ID
        )

        # Impute using helper
        expected_features = _faster_knn_impute_train_test(
            expected_features, train_ind, test_ind, N_NEIGHBORS
        )

        # Calculate sub-feature arrays for validation
        phys_indices = [i for i, f in enumerate(features["All"]) if f in features["Phys"]]
        ecg_indices = [i for i, f in enumerate(features["All"]) if f in features["ECG"]]
        eeg_indices = [i for i, f in enumerate(features["All"]) if f in features["EEG"]]

        # Get actual output
        actual_features = manager._impute_missing_data(
            features_numpy, labels_numpy, metadata, impute_path,
            save_impute=False, load_impute=False,
            num_splits=NUM_SPLITS, kfold_ID=KFOLD_ID, n_neighbors=N_NEIGHBORS
        )

        # Verify results
        assert np.array_equal(actual_features, expected_features, equal_nan=True), \
            "The imputed data does not match expected values."
        assert np.array_equal(actual_features[:, phys_indices], expected_features[:, phys_indices], equal_nan=True), \
            "The physiological features after imputation do not match."
        assert np.array_equal(actual_features[:, ecg_indices], expected_features[:, ecg_indices], equal_nan=True), \
            "The ECG features after imputation do not match."
        assert np.array_equal(actual_features[:, eeg_indices], expected_features[:, eeg_indices], equal_nan=True), \
            "The EEG features after imputation do not match."
        
        # Cleanup temporary file if created
        if os.path.exists(impute_path):
            os.remove(impute_path)
        
        del data_copy, labels, features_numpy, labels_numpy, expected_features
        gc.collect()

    # Used to verify that the imputation function produces the same output as a previously saved imputed dataset (e.g. from a previous run or from a different implementation)
    # def test_same_imputed_data(self, manager, file_paths, gloc_data, gloc_data_imputed_tuple):
    #     gloc_data = gloc_data.copy()
    #     expected_gloc_data_imputed = gloc_data_imputed_tuple[0].copy()

    #     impute_path = "test_imputation.pkl"
    #     num_splits = 5
    #     kfold_ID = 0
    #     n_neighbors = 5
    #     save_impute = False
    #     load_impute = False

    #     feature_groups_to_analyze = FEATURE_GROUPS_EXPLICIT
    #     model_type = ("Complete", "Explicit")
    #     gloc_data, features = manager._process_and_get_feature_names(gloc_data, feature_groups_to_analyze, model_type, file_paths)
    #     gloc_labels = manager._label_gloc_events(gloc_data)
    #     if model_type[0] != "Complete":
    #         gloc_data, gloc_labels = manager._afe_subset(gloc_data, gloc_labels)
    #     gloc_data = manager._reduce_memory(gloc_data, model_type, features)

    #     # Get actual output
    #     gloc_data = manager._impute_missing_data(gloc_data, gloc_labels, impute_path, save_impute, load_impute, num_splits, kfold_ID, n_neighbors)

    #     pd.testing.assert_frame_equal(gloc_data, expected_gloc_data_imputed)
    #     assert gloc_data.equals(expected_gloc_data_imputed), "The imputed gloc_data does not match the expected imputed gloc_data."

    def test_baseline_data(self, manager, file_paths, gloc_data_imputed_tuple):
        def baseline_data(baseline_methods_to_use, trial_column, time_column, event_validated_column, subject_column, features, all_features, gloc, baseline_window,
                        features_phys, all_features_phys, features_ecg, all_features_ecg, features_eeg, all_features_eeg,
                        baseline_data_filename, list_of_baseline_eeg_processed_files, model_type):

            """
            Baselines pre-feature data based on 'baseline_methods_to_use'
            """

            baseline, baseline_derivative, baseline_second_derivative, baseline_names = {}, {}, {}, {}

            # Load participant baseline data once if needed
            participant_seated_rhr = None
            if any(method in baseline_methods_to_use for method in ['v5', 'v6']):
                participant_baseline = pd.read_csv(baseline_data_filename)
                participant_seated_rhr = participant_baseline['resting HR [seated]'][:-1]
                participant_seated_rhr.index = [f"{i:02d}" for i in range(1, 14)]  # Index from '01' to '13'

            # Load EEG baseline data once if needed
            eeg_baseline_data = {}
            if any(method in baseline_methods_to_use for method in ['v7', 'v8']) and 'noAFE' in model_type or 'complete' in model_type:
                eeg_labels = ['delta', 'theta', 'alpha', 'beta']
                for idx, label in enumerate(eeg_labels):
                    eeg_baseline_data[label] = pd.read_csv(list_of_baseline_eeg_processed_files[idx])
                    eeg_baseline_data[label].index = [f"{i:02d}" for i in range(1, 14)]

            # Define baseline functions
            baseline_methods = {
                'v0': lambda: create_v0_baseline(trial_column, time_column, features, all_features),
                'v1': lambda: create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys),
                'v2': lambda: create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys),
                'v3': lambda: create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys),
                'v4': lambda: create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys),
                'v5': lambda: create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg),
                'v6': lambda: create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg),
                'v7': lambda: create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                                eeg_baseline_data['delta'], eeg_baseline_data['theta'],
                                                eeg_baseline_data['alpha'], eeg_baseline_data['beta'], all_features_eeg)
                        if 'noAFE' in model_type or 'complete' in model_type else warnings.warn('EEG baseline methods not implemented for AFE conditions yet.'),
                'v8': lambda: create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg,
                                                eeg_baseline_data['delta'], eeg_baseline_data['theta'],
                                                eeg_baseline_data['alpha'], eeg_baseline_data['beta'], all_features_eeg)
                        if 'noAFE' in model_type or 'complete' in model_type  else warnings.warn('EEG baseline methods not implemented for AFE conditions yet.')
            }

            # Process only selected methods
            for method in baseline_methods_to_use:
                if method in baseline_methods:
                    (baseline[method], baseline_derivative[method], baseline_second_derivative[method],
                    baseline_names[method]) = baseline_methods[method]()

            # Combine all baseline methods into a large dictionary
            combined_baseline, combined_baseline_names = combine_all_baseline(trial_column, baseline, baseline_derivative,
                                                                            baseline_second_derivative, baseline_names)

            return combined_baseline, combined_baseline_names, baseline['v0'], baseline_names['v0']
        
        def create_v0_baseline(trial_column, time_column, features, all_features):
            """
            This function baselines the features with baseline method v0 (no baseline)
            and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """
            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v0 = dict()
            baseline_v0_derivative = dict()
            baseline_v0_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):
                # Find time window
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)

                # Access all features for that trial id
                baseline_v0[trial_id_in_data[i]] = np.array(features[(trial_column == trial_id_in_data[i])])

                # Compute derivative
                time = np.array(time_array[time_index]) 
                baseline_v0_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0[trial_id_in_data[i]], time, axis=0)

                # Compute 2nd derivative
                baseline_v0_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v0_derivative[trial_id_in_data[i]], time, axis=0)

                # Update baseline names
                all_features_updated = [s + '_v0' for s in all_features]

            return baseline_v0, baseline_v0_derivative, baseline_v0_second_derivative, all_features_updated

        def create_v1_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys):
            """
            This function baselines the features with baseline method v1 (divide by mean)
            and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v1 = dict()
            baseline_v1_derivative = dict()
            baseline_v1_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Find time window
                index_window = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)

                # Find baseline average based on specified baseline window
                # baseline_feature = np.mean(features_phys[index_window], axis = 0)
                #
                # # Ensure that data doesn't introduce NaNs.
                # if baseline_feature.shape[0] == 0:
                #     # If baseline window is empty, use ones
                #     baseline_feature = np.ones(features_phys.shape[1])
                # baseline_feature = np.nan_to_num(baseline_feature,nan=1)
                # baseline_feature = np.where(baseline_feature == 0, 1, baseline_feature)

                # Find baseline average based on specified baseline window
                # Ensure that data doesn't introduce NaNs.
                if np.sum(index_window) == 0:
                    # Empty baseline window — use ones
                    baseline_feature = np.ones(features_phys.shape[1])
                else:
                    # Compute mean baseline
                    baseline_feature = np.mean(features_phys[index_window], axis=0)
                    baseline_feature = np.nan_to_num(baseline_feature, nan=1)
                    baseline_feature = np.where(baseline_feature == 0, 1, baseline_feature)

                # Divide features for that trial by baselined data
                baseline_v1[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] /baseline_feature)

                # Compute derivative
                time = time_array[time_index]
                baseline_v1_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v1_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v1_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_phys_updated = [s + '_v1' for s in all_features_phys]

            return baseline_v1, baseline_v1_derivative, baseline_v1_second_derivative, all_features_phys_updated

        def create_v2_baseline(baseline_window, trial_column, time_column, features_phys, all_features_phys):
            """
            This function baselines the features with baseline method v2 (subtract mean)
            and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v2 = dict()
            baseline_v2_derivative = dict()
            baseline_v2_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Find time window
                index_window = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)

                # Find baseline average based on specified baseline window
                baseline_feature = np.mean(features_phys[index_window], axis = 0)
                baseline_feature = np.nan_to_num(baseline_feature, nan=0)

                # Subtract baseline data (baseline v2)
                baseline_v2[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] - baseline_feature)

                # Compute derivative
                time = time_array[time_index]
                baseline_v2_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v2_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v2_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_phys_updated = [s + '_v2' for s in all_features_phys]

            return baseline_v2, baseline_v2_derivative, baseline_v2_second_derivative, all_features_phys_updated

        def create_v3_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys):
            """
            This function baselines the features with baseline method v3 (Each feature prior to ROR
            is divided by the first portion of the trial for baseline & the rest of the
            trial is divided by the portion prior to ROR for baseline) and puts them into a
            dictionary per trial. This dictionary is output. The first and second derivative of
            each baselined feature are computed and output in a dictionary.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v3 = dict()
            baseline_v3_derivative = dict()
            baseline_v3_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Define first baseline period (during initial specified baseline window)
                index_window1 = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))

                # Find indices of other variables
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)
                time = time_array[time_index]
                event_array = np.array(event_validated_column)

                # Find baseline average based on specified baseline window
                baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)

                # If ROR marked in trial, find when ROR begins and take the baseline directly before that (with window size defined in main)
                current_trial_event = event_array[time_index]
                if 'begin ROR' in current_trial_event:
                    begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
                    time_ROR = time_array[begin_ROR_index][0,0]

                    # Define second baseline period
                    index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR) &
                                    (trial_column == trial_id_in_data[i]))

                    # Find baseline average based on specified baseline window
                    baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

                    # Divide features for that trial by correct baseline period
                    first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
                    second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])

                    # Stack data from two windows of baselined data together
                    baseline_v3[trial_id_in_data[i]] = np.array(features_phys[first_baseline_period] / baseline_period1_feature)
                    baseline_v3[trial_id_in_data[i]] = np.vstack((baseline_v3[trial_id_in_data[i]], np.array(features_phys[second_baseline_period] / baseline_period2_feature)))

                # if no labeled begin of ROR, then just use the intial baseline period
                else:
                    baseline_v3[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] / baseline_period1_feature)


                # Compute derivative
                baseline_v3_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v3_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v3_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_phys_updated = [s + '_v3' for s in all_features_phys]

            return baseline_v3, baseline_v3_derivative, baseline_v3_second_derivative, all_features_phys_updated

        def create_v4_baseline(baseline_window, trial_column, time_column, event_validated_column, features_phys, all_features_phys):
            """
            This function baselines the features with baseline method v4 (The first portion of
            the trial is subtracted from each feature prior to ROR & the period prior to ROR gets
            subtracted from the feature for the rest of the trial) and puts them into a
            dictionary per trial. This dictionary is output. The first and second derivative of
            each baselined feature are computed and output in a dictionary.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v4 = dict()
            baseline_v4_derivative = dict()
            baseline_v4_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Define first baseline period (during initial specified baseline window)
                index_window1 = ((time_column<baseline_window) & (trial_column == trial_id_in_data[i]))

                # Find indices of other variables
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)
                time = time_array[time_index]
                event_array = np.array(event_validated_column)

                # Find baseline average based on specified baseline window
                baseline_period1_feature = np.mean(features_phys[index_window1], axis=0)

                # If ROR marked in trial, find when ROR begins and take the baseline directly before that (with window size defined in main)
                current_trial_event = event_array[time_index]
                if 'begin ROR' in current_trial_event:
                    begin_ROR_index = np.argwhere(current_trial_event == 'begin ROR')
                    time_ROR = time_array[begin_ROR_index][0,0]

                    # Define second baseline period
                    index_window2 = ((time_column>(time_ROR - baseline_window)) & (time_column<time_ROR)
                                    & (trial_column == trial_id_in_data[i]))

                    # Find baseline average based on specified baseline window
                    baseline_period2_feature = np.mean(features_phys[index_window2], axis=0)

                    # Subtract features for that trial by correct baseline period
                    first_baseline_period = (time_array < (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])
                    second_baseline_period = (time_array >= (time_ROR - baseline_window)) & (trial_column == trial_id_in_data[i])

                    # Stack data from two windows of baselined data together
                    baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(first_baseline_period)] - baseline_period1_feature)
                    baseline_v4[trial_id_in_data[i]] = np.vstack((baseline_v4[trial_id_in_data[i]], np.array(features_phys[(second_baseline_period)] - baseline_period2_feature)))

                # if no labeled begin of ROR, then just use the intial baseline period
                else:
                    baseline_v4[trial_id_in_data[i]] = np.array(features_phys[(trial_column == trial_id_in_data[i])] / baseline_period1_feature)


                # Compute derivative
                baseline_v4_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v4_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v4_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_phys_updated = [s + '_v4' for s in all_features_phys]

            return baseline_v4, baseline_v4_derivative, baseline_v4_second_derivative, all_features_phys_updated

        def create_v5_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg):
            """
            This function baselines the features with baseline method v5 (divide by RHR
            from another study) and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v5 = dict()
            baseline_v5_derivative = dict()
            baseline_v5_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Find time index and current participant
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)
                participant_array = np.array(subject_column)

                # Find baseline for current participant
                current_participant = participant_array[time_index][0]
                baseline_feature = participant_seated_rhr[current_participant]
                baseline_feature = np.nan_to_num(baseline_feature, nan=1)

                # Divide features for that trial by baselined data
                baseline_v5[trial_id_in_data[i]] = np.array(features_ecg[(trial_column == trial_id_in_data[i])] /baseline_feature)

                # Compute derivative
                time = time_array[time_index]
                baseline_v5_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v5_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v5_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_ecg_updated = [s + '_v5' for s in all_features_ecg]

            return baseline_v5, baseline_v5_derivative, baseline_v5_second_derivative, all_features_ecg_updated

        def create_v6_baseline(baseline_window, trial_column, time_column, subject_column, features_ecg, participant_seated_rhr, all_features_ecg):
            """
            This function baselines the features with baseline method v6 (subtract RHR
            from another study) and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v6 = dict()
            baseline_v6_derivative = dict()
            baseline_v6_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Find time index and current participant
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)
                participant_array = np.array(subject_column)

                # Find baseline for current participant
                current_participant = participant_array[time_index][0]
                baseline_feature = participant_seated_rhr[current_participant]
                baseline_feature = np.nan_to_num(baseline_feature, nan=0)

                # Divide features for that trial by baselined data
                baseline_v6[trial_id_in_data[i]] = np.array(features_ecg[(trial_column == trial_id_in_data[i])] - baseline_feature)

                # Compute derivative
                time = time_array[time_index]
                baseline_v6_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v6_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v6_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_ecg_updated = [s + '_v6' for s in all_features_ecg]

            return baseline_v6, baseline_v6_derivative, baseline_v6_second_derivative, all_features_ecg_updated

        def create_v7_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                            eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
            """
            This function baselines the features with baseline method v7 (divide by processed EEG baseline)
            and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """
            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v7 = dict()
            baseline_v7_derivative = dict()
            baseline_v7_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Find time index and current participant
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)
                participant_array = np.array(subject_column)

                # Find baseline for current participant
                current_participant = participant_array[time_index][0]
                baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
                baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
                baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
                baseline_feature_beta = eeg_baseline_beta.loc[current_participant]

                # Set current trial
                current_trial_data = features_eeg[time_index]

                # Initialize np array to fill with each trial's data
                baselined_trial_data = np.zeros_like(current_trial_data)

                # Iterate through columns to divide by correct baseline channel
                for col in range(len(all_features_eeg)):
                    # Get current column name
                    column_name = all_features_eeg[col]

                    # From column name in all_features_eeg, obtain the corresponding column name in
                    # the baseline CSV file
                    name_split = column_name.split('_')
                    channel_name = name_split[0]

                    # Find index of corresponding channel name in baseline file
                    channel_index = baseline_feature_delta.index.get_loc(channel_name)

                    if name_split[1] == 'delta - EEG':
                        # Divide features for that trial by baselined data
                        if baseline_feature_delta.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:,col] = current_trial_data[:,col] / baseline_feature_delta.iloc[channel_index]
                    elif name_split[1] == 'theta - EEG':
                        # Divide features for that trial by baselined data
                        if baseline_feature_theta.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_theta.iloc[channel_index]
                    elif name_split[1] == 'alpha - EEG':
                        # Divide features for that trial by baselined data
                        if baseline_feature_alpha.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_alpha.iloc[channel_index]
                    elif name_split[1] == 'beta - EEG':
                        # Divide features for that trial by baselined data
                        if baseline_feature_beta.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:, col] = current_trial_data[:, col] / baseline_feature_beta.iloc[channel_index]

                # Once all columns have been iterated through, define baseline_v7
                baseline_v7[trial_id_in_data[i]] = baselined_trial_data

                # Compute derivative
                time = time_array[time_index]
                baseline_v7_derivative[trial_id_in_data[i]] = np.gradient(baseline_v7[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v7_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v7_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_eeg_updated = [s + '_v7' for s in all_features_eeg]

            return baseline_v7, baseline_v7_derivative, baseline_v7_second_derivative, all_features_eeg_updated

        def create_v8_baseline(baseline_window, trial_column, time_column, subject_column, features_eeg, eeg_baseline_delta,
                            eeg_baseline_theta, eeg_baseline_alpha, eeg_baseline_beta, all_features_eeg):
            """
            This function baselines the features with baseline method v8 (subtract EEG baseline)
            and puts them into a dictionary per trial. This dictionary is output.
            The first and second derivative of each baselined feature are computed and
            output in a dictionary.
            """
            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Build Dictionary for each trial_id
            baseline_v8 = dict()
            baseline_v8_derivative = dict()
            baseline_v8_second_derivative = dict()

            # Iterate through trials
            for i in range(np.size(trial_id_in_data)):

                # Find time index and current participant
                time_index = (trial_column == trial_id_in_data[i])
                time_array = np.array(time_column)
                participant_array = np.array(subject_column)

                # Find baseline for current participant
                current_participant = participant_array[time_index][0]
                baseline_feature_delta = eeg_baseline_delta.loc[current_participant]
                baseline_feature_theta = eeg_baseline_theta.loc[current_participant]
                baseline_feature_alpha = eeg_baseline_alpha.loc[current_participant]
                baseline_feature_beta = eeg_baseline_beta.loc[current_participant]

                # Set current trial
                current_trial_data = features_eeg[time_index]

                # Initialize np array to fill with each trial's data
                baselined_trial_data = np.zeros_like(current_trial_data)

                # Iterate through columns to divide by correct baseline channel
                for col in range(len(all_features_eeg)):
                    # Get current column name
                    column_name = all_features_eeg[col]

                    # From column name in all_features_eeg, obtain the corresponding column name in
                    # the baseline CSV file
                    name_split = column_name.split('_')
                    channel_name = name_split[0]

                    # Find index of corresponding channel name in baseline file
                    channel_index = baseline_feature_delta.index.get_loc(channel_name)

                    if name_split[1] == 'delta - EEG':
                        # Subtract features for that trial by baselined data
                        if baseline_feature_delta.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:,col] = current_trial_data[:,col] - baseline_feature_delta.iloc[channel_index]
                    elif name_split[1] == 'theta - EEG':
                        # Subtract features for that trial by baselined data
                        if baseline_feature_theta.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_theta.iloc[channel_index]
                    elif name_split[1] == 'alpha - EEG':
                        # Subtract features for that trial by baselined data
                        if baseline_feature_alpha.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_alpha.iloc[channel_index]
                    elif name_split[1] == 'beta - EEG':
                        # Subtract features for that trial by baselined data
                        if baseline_feature_beta.iloc[channel_index] == 0:
                            baselined_trial_data[:, col] = np.zeros(np.shape(baselined_trial_data)[0])
                        else:
                            baselined_trial_data[:, col] = current_trial_data[:, col] - baseline_feature_beta.iloc[channel_index]

                # Once all columns have been iterated through, define baseline_v7
                baseline_v8[trial_id_in_data[i]] = baselined_trial_data

                # Compute derivative
                time = time_array[time_index]
                baseline_v8_derivative[trial_id_in_data[i]] = np.gradient(baseline_v8[trial_id_in_data[i]], time, axis = 0)

                # Compute 2nd derivative
                baseline_v8_second_derivative[trial_id_in_data[i]] = np.gradient(baseline_v8_derivative[trial_id_in_data[i]], time, axis = 0)

                # Update baseline names
                all_features_eeg_updated = [s + '_v8' for s in all_features_eeg]

            return baseline_v8, baseline_v8_derivative, baseline_v8_second_derivative, all_features_eeg_updated

        def combine_all_baseline(trial_column, baseline, baseline_derivative, baseline_second_derivative, baseline_names):
            """
            This function combines the features, derivative of features, and second derivative of features into one np array.
            """

            # Find Unique Trial ID
            trial_id_in_data = np.unique(trial_column)

            # Preallocate the dictionary with NumPy arrays
            num_cols = 0
            for method in baseline.keys():
                num_cols += baseline[method][trial_id_in_data[0]].shape[1]*3
            combined_baseline = {trial: np.empty((baseline[list(baseline.keys())[0]][trial].shape[0], num_cols), dtype=np.float32) for trial in trial_id_in_data}

            # Iterate through all unique trial_id & combine the baseline, baseline derivative, and baseline second derivative
            for trial in trial_id_in_data:
                all_data = []
                for method in baseline.keys():
                    base = baseline[method][trial].astype(np.float32)
                    deriv = baseline_derivative[method][trial].astype(np.float32)
                    second = baseline_second_derivative[method][trial].astype(np.float32)
                    all_data.append(np.hstack([base, deriv, second]))

                combined_baseline[trial] = np.hstack(all_data).astype(np.float32)


            combined_baseline_names = sum([baseline_names[method] + [s + '_derivative' for s in baseline_names[method]] +
                                            [s + '_2derivative' for s in baseline_names[method]] for method in baseline_names.keys()], [])

            return combined_baseline, combined_baseline_names

        # Setup Data
        gloc_data_all_features_imputed_numpy = gloc_data_imputed_tuple[0].copy()
        gloc_labels_numpy = gloc_data_imputed_tuple[1].copy()
        features = gloc_data_imputed_tuple[2].copy()
        experiment_metadata = gloc_data_imputed_tuple[3].copy()

        baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"]
        model_type = ("Complete", "Explicit")
        baseline_window = 32.5

        # Get column indices for respective feature groups
        phys_indices = [i for i, feature in enumerate(features["All"]) if feature in features["Phys"]]
        ecg_indices = [i for i, feature in enumerate(features["All"]) if feature in features["ECG"]]
        eeg_indices = [i for i, feature in enumerate(features["All"]) if feature in features["EEG"]]

        # Create expected copy
        expected_gloc_data_all_features_imputed_numpy, expected_gloc_labels_numpy = gloc_data_all_features_imputed_numpy.copy(), gloc_labels_numpy.copy()
        expected_combined_baseline, expected_combined_baseline_names, baseline_v0, baseline_names_v0 = (
            baseline_data(
                baseline_methods_to_use, 
                experiment_metadata["trial_id"], 
                experiment_metadata["Time (s)"], 
                experiment_metadata["event_validated"], 
                experiment_metadata["subject"],
                expected_gloc_data_all_features_imputed_numpy, 
                features["All"],
                expected_gloc_labels_numpy,
                baseline_window, 
                expected_gloc_data_all_features_imputed_numpy[:, phys_indices], 
                features["Phys"], 
                expected_gloc_data_all_features_imputed_numpy[:, ecg_indices],
                features["ECG"],
                expected_gloc_data_all_features_imputed_numpy[:, eeg_indices], 
                features["EEG"], 
                file_paths["baseline"], 
                file_paths["baseline_eeg_processed_list"],
                model_type))

        # Get actual output
        combined_baseline, combined_baseline_names = manager._get_combined_baseline_data(gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window, baseline_methods_to_use, features, file_paths, model_type)

        assert all(np.array_equal(expected_combined_baseline[trial_id], combined_baseline[trial_id]) for trial_id in expected_combined_baseline.keys()), "Combined baseline data does not match expected data."
        assert np.array_equal(expected_combined_baseline_names, combined_baseline_names), "Combined baseline names do not match expected names."

    def test_generate_features(self, manager, gloc_data_imputed_tuple, file_paths):
        def pull_unengineered_streams():
            # Create Raw Feature Indices
            unengineered_streams = ['HR (bpm) - Equivital',
                                    'ECG Lead 1 - Equivital', 'ECG Lead 2 - Equivital',
                                    'HR_instant - Equivital', 'HR_average - Equivital', 'HR_w_average - Equivital',
                                    'BR (rpm) - Equivital',
                                    'Skin Temperature - IR Thermometer (°C) - Equivital',

                                    'Pupil position left X [HUCS mm] - Tobii', 'Pupil position left Y [HUCS mm] - Tobii',
                                    'Pupil position left Z [HUCS mm] - Tobii', 'Pupil position right X [HUCS mm] - Tobii',
                                    'Pupil position right Y [HUCS mm] - Tobii', 'Pupil position right Z [HUCS mm] - Tobii',
                                    'Pupil diameter left [mm] - Tobii', 'Pupil diameter right [mm] - Tobii',

                                    'F1 - EEG', 'Fz - EEG', 'F3 - EEG', 'C3 - EEG', 'C4 - EEG', 'CP1 - EEG', 'CP2 - EEG',
                                    'T8 - EEG', 'TP9 - EEG', 'TP10 - EEG', 'P7 - EEG', 'P8 - EEG', 'AFz - EEG', 'AF4 - EEG',
                                    'FT9 - EEG', 'FT10 - EEG', 'FC5 - EEG', 'FC3 - EEG', 'FC1 - EEG', 'FC2 - EEG', 'FC4 - EEG',
                                    'FC6 - EEG', 'C5 - EEG', 'Cz - EEG', 'CP5 - EEG', 'CP6 - EEG', 'P5 - EEG', 'P3 - EEG',
                                    'P1 - EEG', 'Pz - EEG', 'P4 - EEG', 'P6 - EEG',

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

                                    'F1_delta - EEG', 'F1_theta - EEG', 'F1_alpha - EEG', 'F1_beta - EEG',
                                    'Fz_delta - EEG', 'Fz_theta - EEG', 'Fz_alpha - EEG', 'Fz_beta - EEG',
                                    'F3_delta - EEG', 'F3_theta - EEG', 'F3_alpha - EEG', 'F3_beta - EEG',
                                    'C3_delta - EEG', 'C3_theta - EEG', 'C3_alpha - EEG', 'C3_beta - EEG',
                                    'C4_delta - EEG', 'C4_theta - EEG', 'C4_alpha - EEG', 'C4_beta - EEG',
                                    'CP1_delta - EEG', 'CP1_theta - EEG', 'CP1_alpha - EEG', 'CP1_beta - EEG',
                                    'CP2_delta - EEG', 'CP2_theta - EEG', 'CP2_alpha - EEG', 'CP2_beta - EEG',
                                    'T8_delta - EEG', 'T8_theta - EEG', 'T8_alpha - EEG', 'T8_beta - EEG',
                                    'TP9_delta - EEG', 'TP9_theta - EEG', 'TP9_alpha - EEG', 'TP9_beta - EEG',
                                    'TP10_delta - EEG', 'TP10_theta - EEG', 'TP10_alpha - EEG', 'TP10_beta - EEG',
                                    'P7_delta - EEG', 'P7_theta - EEG', 'P7_alpha - EEG', 'P7_beta - EEG',
                                    'P8_delta - EEG', 'P8_theta - EEG', 'P8_alpha - EEG', 'P8_beta - EEG',
                                    'AFz_delta - EEG', 'AFz_theta - EEG', 'AFz_alpha - EEG', 'AFz_beta - EEG',
                                    'AF4_delta - EEG', 'AF4_theta - EEG', 'AF4_alpha - EEG', 'AF4_beta - EEG',
                                    'FT9_delta - EEG', 'FT9_theta - EEG', 'FT9_alpha - EEG', 'FT9_beta - EEG',
                                    'FT10_delta - EEG', 'FT10_theta - EEG', 'FT10_alpha - EEG', 'FT10_beta - EEG',
                                    'FC5_delta - EEG', 'FC5_theta - EEG', 'FC5_alpha - EEG', 'FC5_beta - EEG',
                                    'FC3_delta - EEG', 'FC3_theta - EEG', 'FC3_alpha - EEG', 'FC3_beta - EEG',
                                    'FC1_delta - EEG', 'FC1_theta - EEG', 'FC1_alpha - EEG', 'FC1_beta - EEG',
                                    'FC2_delta - EEG', 'FC2_theta - EEG', 'FC2_alpha - EEG', 'FC2_beta - EEG',
                                    'FC4_delta - EEG', 'FC4_theta - EEG', 'FC4_alpha - EEG', 'FC4_beta - EEG',
                                    'FC6_delta - EEG', 'FC6_theta - EEG', 'FC6_alpha - EEG', 'FC6_beta - EEG',
                                    'C5_delta - EEG', 'C5_theta - EEG', 'C5_alpha - EEG', 'C5_beta - EEG',
                                    'Cz_delta - EEG', 'Cz_theta - EEG', 'Cz_alpha - EEG', 'Cz_beta - EEG',
                                    'CP5_delta - EEG', 'CP5_theta - EEG', 'CP5_alpha - EEG', 'CP5_beta - EEG',
                                    'CP6_delta - EEG', 'CP6_theta - EEG', 'CP6_alpha - EEG', 'CP6_beta - EEG',
                                    'P5_delta - EEG', 'P5_theta - EEG', 'P5_alpha - EEG', 'P5_beta - EEG',
                                    'P3_delta - EEG', 'P3_theta - EEG', 'P3_alpha - EEG', 'P3_beta - EEG',
                                    'P1_delta - EEG', 'P1_theta - EEG', 'P1_alpha - EEG', 'P1_beta - EEG',
                                    'Pz_delta - EEG', 'Pz_theta - EEG', 'Pz_alpha - EEG', 'Pz_beta - EEG',
                                    'P4_delta - EEG', 'P4_theta - EEG', 'P4_alpha - EEG', 'P4_beta - EEG',
                                    'P6_delta - EEG', 'P6_theta - EEG', 'P6_alpha - EEG', 'P6_beta - EEG']

            return unengineered_streams

        def convert_to_unique_ordered_integers(strings):
            mapping = {}
            result = []
            current_id = 1
            for s in strings:
                if s not in mapping:
                    mapping[s] = current_id
                    current_id += 1
                result.append(mapping[s])

            return np.array(result,dtype=np.float32)
        
        # Setup data
        gloc_data_all_features_imputed_numpy = gloc_data_imputed_tuple[0].copy()
        gloc_labels_numpy = gloc_data_imputed_tuple[1].copy()
        features = gloc_data_imputed_tuple[2].copy()
        experiment_metadata = gloc_data_imputed_tuple[3].copy()

        baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"]
        model_type = ("Complete", "Explicit")
        baseline_window = 32.5

        combined_baseline, combined_baseline_names = manager._get_combined_baseline_data(gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window, baseline_methods_to_use, features, file_paths, model_type)
        


        # Getting expected returns
        expected_combined_baseline, expected_combined_baseline_names = combined_baseline.copy(), combined_baseline_names.copy()
        # Unpack without feature generation
        expected_x_feature_matrix = np.vstack([expected_combined_baseline[trial_id] for trial_id in expected_combined_baseline]).astype(np.float32)

        # Only grab unengineered datastreams
        unengineered_streams = pull_unengineered_streams()

        # Grab indices corresponding to unengineered features in unengineered streams (but also with baseline suffix id)
        ue_indices = [
            i for i, feature in enumerate(expected_combined_baseline_names)
            if (
                    feature in unengineered_streams
                    or any(
                f"{stream}_{suffix}" == feature for stream in unengineered_streams for suffix in baseline_methods_to_use)
            )
        ]

        # Get new x_feature matrix
        expected_x_feature_matrix = expected_x_feature_matrix[:, ue_indices]
        expected_trial_ints = convert_to_unique_ordered_integers(experiment_metadata["trial_id"])

        expected_x_feature_matrix = np.hstack([expected_x_feature_matrix, expected_trial_ints.reshape(-1, 1)])
        expected_y_gloc_labels = gloc_labels_numpy.copy()

        expected_all_features = expected_combined_baseline_names
        expected_all_features = [expected_all_features[i] for i in ue_indices]



        # Get actual returns
        x_feature_matrix, all_features = manager._generate_features(baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata)
        features["All"] = all_features



        assert np.array_equal(expected_x_feature_matrix, x_feature_matrix), "X feature matrix does not match expected matrix."
        assert np.array_equal(expected_all_features, features["All"]), "All features list does not match expected list."
        assert np.array_equal(expected_y_gloc_labels, gloc_labels_numpy), "GLOC labels do not match expected labels."
        assert np.array_equal(expected_trial_ints, experiment_metadata["trial_ints"]), "Trial IDs in metadata do not match expected trial IDs."

    def test_feature_clean_and_prep(self, manager, gloc_data_imputed_tuple, file_paths):
        def remove_constant_columns(x_feature_matrix_noNaN, all_features):
            """
            This function removes all constant columns before feeding into the ML classifiers.
            """
            # Find all constant columns
            constant_columns = np.all(x_feature_matrix_noNaN == x_feature_matrix_noNaN[0,:], axis = 0)

            # Remove all constant columns from data frame
            x_feature_matrix_noNaN = x_feature_matrix_noNaN[:, ~constant_columns]

            all_features = [all_features[i] for i in range(len(all_features)) if ~constant_columns[i]]

            return x_feature_matrix_noNaN, all_features

        def process_NaN(y_gloc_labels, x_feature_matrix, all_features, trials):
            """
            This is a temporary function for removing all rows with NaN values. This can be replaced by
            another method in the future, but is necessary for feeding into ML Classifiers.
            """
            # Find & remove columns if they have all NaN values
            nan_test = np.isnan(x_feature_matrix)
            index_column_all_NaN = np.all(nan_test, axis=0)
            x_feature_matrix_noNaN_cols = x_feature_matrix[:, ~index_column_all_NaN]

            # Adjust all_features to only include columns that don't have all NaN
            all_features = [all_features[i] for i in range(len(all_features)) if ~index_column_all_NaN[i]]

            # Find & Remove rows in label array if they have NaN values
            y_gloc_labels_noNaN = y_gloc_labels[~np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)]

            # Find & Remove rows in trial array if they have NaN values
            trials_noNaN = trials[~np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)]

            # Find & Remove rows in X matrix if the features have any NaN values in that row
            x_feature_matrix_noNaN = x_feature_matrix_noNaN_cols[~np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)]

            return y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, trials_noNaN

        # Setup data
        gloc_data_all_features_imputed_numpy = gloc_data_imputed_tuple[0].copy()
        gloc_labels_numpy = gloc_data_imputed_tuple[1].copy()
        features = gloc_data_imputed_tuple[2].copy()
        experiment_metadata = gloc_data_imputed_tuple[3].copy()

        baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"]
        model_type = ("Complete", "Explicit")
        baseline_window = 32.5
        impute_type = 1

        combined_baseline, combined_baseline_names = manager._get_combined_baseline_data(gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window, baseline_methods_to_use, features, file_paths, model_type)
        x_feature_matrix, features["All"] = manager._generate_features(baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata)

        expected_x_feature_matrix, expected_all_features = x_feature_matrix.copy(), features["All"].copy()
        expected_y_gloc_labels = gloc_labels_numpy.copy()
        # Remove constant columns (typically no constant columns)
        expected_x_feature_matrix, expected_all_features = remove_constant_columns(expected_x_feature_matrix, expected_all_features)

        # Add back in as 2nd to last column for explicit only
        # (needs to be 2nd to last for advanced pipeline - could be last for traditional)
        model_kind, label_mode = model_type
        if model_kind == "Complete" and label_mode == "Explicit":
            expected_x_feature_matrix = np.hstack([
                expected_x_feature_matrix[:, :-1],
                experiment_metadata["AFE_indicator"].reshape(-1, 1),
                expected_x_feature_matrix[:, -1:]
            ])

        # List-wise deletion or clean any residual NaNs
        if impute_type == 2 or impute_type == 1:
            # Remove rows with NaN (temporary solution-should replace with other method eventually)
            expected_y_gloc_labels_noNaN, expected_x_feature_matrix_noNaN, expected_all_features, expected_trials_noNaN = process_NaN(expected_y_gloc_labels,
                                                                                                expected_x_feature_matrix,
                                                                                                expected_all_features, experiment_metadata["trial_ints"])
        else:
            expected_y_gloc_labels_noNaN, expected_x_feature_matrix_noNaN, expected_trials_noNaN = expected_y_gloc_labels, expected_x_feature_matrix, experiment_metadata["trial_ints"]

        # Get actual returns
        x_feature_matrix, y_gloc_labels, features["All"], experiment_metadata["trial_ints"] = manager._feature_clean_and_prep(x_feature_matrix, gloc_labels_numpy, features, experiment_metadata, model_type, impute_type)

        assert np.array_equal(expected_x_feature_matrix_noNaN, x_feature_matrix), "Cleaned X feature matrix does not match expected matrix."
        assert np.array_equal(expected_y_gloc_labels_noNaN, y_gloc_labels), "Cleaned GLOC labels do not match expected labels."
        assert np.array_equal(expected_all_features, features["All"]), "Cleaned all features list does not match expected list."
        assert np.array_equal(expected_trials_noNaN, experiment_metadata["trial_ints"]), "Cleaned trial IDs in metadata do not match expected trial IDs."

    def test_train_test_split(self, manager, gloc_data_imputed_tuple, file_paths):
        def groupedtrial_kfold_split(Y, X, trials, num_splits, kfold_ID):
            """
            This function splits the X and y matrix into training and test matrix.
            """

            # Grouped K-Fold setup
            # Use random state to ensure repeatability across runs and classifiers
            gkf = StratifiedGroupKFold(n_splits=num_splits, shuffle=False)

            # Safety check to ensure that kfold_ID is within the fold indices
            n_folds = gkf.get_n_splits()
            if kfold_ID < 0 or kfold_ID >= n_folds:
                raise ValueError(f"Fold index {kfold_ID} out of range (must be between 0 and {n_folds - 1})")

            # Grab train and test indices given the skf generator format for a specific kfold_ID
            train_index, test_index = next(islice(gkf.split(X, Y, trials), kfold_ID, kfold_ID + 1))

            # Extract the corresponding data for the given kfold_ID
            x_train, y_train = X[train_index], Y[train_index]
            x_test, y_test = X[test_index], Y[test_index]

            return x_train, x_test, y_train, y_test, train_index, test_index
        
        # Setup Data
        gloc_data_all_features_imputed_numpy = gloc_data_imputed_tuple[0].copy()
        gloc_labels_numpy = gloc_data_imputed_tuple[1].copy()
        features = gloc_data_imputed_tuple[2].copy()
        experiment_metadata = gloc_data_imputed_tuple[3].copy()

        baseline_methods_to_use = ["v0", "v1", "v2", "v5", "v6"]
        model_type = ("Complete", "Explicit")
        baseline_window = 32.5
        impute_type = 1
        num_splits = 5
        kfold_ID = 0

        combined_baseline, combined_baseline_names = manager._get_combined_baseline_data(gloc_data_all_features_imputed_numpy, experiment_metadata, baseline_window, baseline_methods_to_use, features, file_paths, model_type)
        x_feature_matrix, features["All"] = manager._generate_features(baseline_methods_to_use, combined_baseline, combined_baseline_names, experiment_metadata)
        x_feature_matrix, y_gloc_labels, features["All"], experiment_metadata["trial_ints"] = manager._feature_clean_and_prep(x_feature_matrix, gloc_labels_numpy, features, experiment_metadata, model_type, impute_type)



        # Get Expected Data
        expected_x_feature_matrix = x_feature_matrix.copy()
        expected_y_gloc_labels = y_gloc_labels.copy()
        expected_trial_ints = experiment_metadata["trial_ints"].copy()

        # Training/Test Split
        expected_x_train, expected_x_test, expected_y_train, expected_y_test, _, _ = groupedtrial_kfold_split(
            expected_y_gloc_labels, expected_x_feature_matrix, expected_trial_ints, num_splits, kfold_ID)

        # Grab trials as separate
        expected_x_train_trials = expected_x_train[:, -1].reshape(-1, 1)
        expected_x_train = expected_x_train[:, :-1]
        expected_x_test_trials = expected_x_test[:, -1].reshape(-1, 1)
        expected_x_test = expected_x_test[:, :-1]

        # And standardize based on training data
        scaler = StandardScaler()
        expected_x_train = scaler.fit_transform(expected_x_train)
        expected_x_test = scaler.transform(expected_x_test)

        # Add indices back as final column
        expected_x_train = np.hstack([expected_x_train, expected_x_train_trials])
        expected_x_test = np.hstack([expected_x_test, expected_x_test_trials])



        # Get Actual Returns
        x_train, y_train, x_test, y_test = manager._get_train_test_split(x_feature_matrix, y_gloc_labels, experiment_metadata, num_splits, kfold_ID)

        assert np.array_equal(expected_x_train, x_train), "Training X matrix does not match expected training X matrix."
        assert np.array_equal(expected_y_train, y_train), "Training Y vector does not match expected training Y vector."
        assert np.array_equal(expected_x_test, x_test), "Test X matrix does not match expected test X matrix."
        assert np.array_equal(expected_y_test, y_test), "Test Y vector does not match expected test Y vector."

    def test_load_and_prepare_data(self):
        """End-to-end test for loading and preparing data."""
        model_type = self.MODEL_TYPE
        # Get the absolute path to the data folder
        test_dir = os.path.dirname(os.path.abspath(__file__))
        datafolder = os.path.join(os.path.dirname(test_dir), "data")
        
        model_str = f"{model_type[0]}_{model_type[1]}"
        num_splits = 5
        kfold_ID = 0
        impute_path = os.path.join(test_dir, "testing_temp", f"gloc_data_imputed_{model_str}_e2e.pkl")
        impute_type = 1
        n_neighbors = 4
        baseline_window = 32.5
        analysis_type = 2
        save_impute = False
        load_impute = False

        # Convert model_type to lowercase for legacy function
        legacy_model_type = (model_type[0].lower() if model_type[0] != "noAFE" else model_type[0], model_type[1].lower())
        
        expected_x_train, expected_x_test, expected_y_train, expected_y_test, expected_all_features = load_and_prepare_data_advanced(
            model_type = legacy_model_type,
            num_splits = num_splits,
            kfold_ID = kfold_ID,
            impute_path = impute_path,
            impute_type = impute_type,
            n_neighbors = n_neighbors,
            baseline_window = baseline_window,
            datafolder = datafolder,
            analysis_type = analysis_type,
            save_impute = save_impute,
            load_impute = load_impute
        )

        data_manager = DataManager(data_path = datafolder, testing = True, use_reduced_dataset = USE_REDUCED_DATASET)
        x_train, x_test, y_train, y_test, all_features = data_manager.get_data(
            model_type = model_type,
            num_splits = num_splits,
            kfold_ID = kfold_ID,
            impute_path = impute_path,
            subject_to_analyze = None,
            trial_to_analyze = None,
            impute_type = impute_type,
            n_neighbors = n_neighbors,
            baseline_window = baseline_window,
            analysis_type = analysis_type,
            remove_NaN_trials = True,
            save_impute = save_impute,
            load_impute = load_impute
        )

        assert np.array_equal(expected_x_train, x_train), "Training X matrix does not match expected training X matrix."
        assert np.array_equal(expected_x_test, x_test), "Test X matrix does not match expected test X matrix."
        assert np.array_equal(expected_y_train, y_train), "Training Y vector does not match expected training Y vector."
        assert np.array_equal(expected_y_test, y_test), "Test Y vector does not match expected test Y vector."
        assert np.array_equal(expected_all_features, all_features), "All features list does not match expected all features list."


# ===== Four Model-Specific Test Classes =====

class TestDataManagerCompleteExplicit(TestDataManagerBase):
    """Tests for Complete/Explicit model type."""
    MODEL_TYPE = ("Complete", "Explicit")
    __test__ = True  # Ensure this class is collected by pytest

class TestDataManagerCompleteImplicit(TestDataManagerBase):
    """Tests for Complete/Implicit model type."""
    MODEL_TYPE = ("Complete", "Implicit")
    __test__ = True  # Ensure this class is collected by pytest

class TestDataManagerNoAFEExplicit(TestDataManagerBase):
    """Tests for noAFE/Explicit model type."""
    MODEL_TYPE = ("noAFE", "Explicit")
    __test__ = True  # Ensure this class is collected by pytest

class TestDataManagerNoAFEImplicit(TestDataManagerBase):
    """Tests for noAFE/Implicit model type."""
    MODEL_TYPE = ("noAFE", "Implicit")
    __test__ = True  # Ensure this class is collected by pytest