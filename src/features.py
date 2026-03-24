from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import warnings

class BaseFeatureGroup(ABC):
    """Interface for all feature groups."""

    @abstractmethod
    def get_feature_names(self, model_type):
        """Returns the list of feature names for the given model type."""
        pass

    def process(self, df, file_paths): 
        """
        Optional: Perform specific manipulations of features in the dataframe.
        Default implementation does nothing.
        Override in subclasses if needed.
        """
        return df
    
# Concrete Implementations
class ECGGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        return ["HR (bpm) - Equivital", "ECG Lead 1 - Equivital", "ECG Lead 2 - Equivital", "HR_instant - Equivital","HR_average - Equivital", "HR_w_average - Equivital"]
    
class BRGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        return ["BR (rpm) - Equivital"]
    
class TempGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        return ["Skin Temperature - IR Thermometer (°C) - Equivital"]
    
class EyeTrackingGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        return [
            "Pupil position left X [HUCS mm] - Tobii", 
            "Pupil position left Y [HUCS mm] - Tobii", 
            "Pupil position left Z [HUCS mm] - Tobii",
            "Pupil position right X [HUCS mm] - Tobii", 
            "Pupil position right Y [HUCS mm] - Tobii", 
            "Pupil position right Z [HUCS mm] - Tobii",
            "Pupil diameter left [mm] - Tobii", 
            "Pupil diameter right [mm] - Tobii", 
            "Pupil Difference [mm]"
        ]
    
    def process(self, df, file_paths):
        # Pupil Difference
        pupil_difference = df["Pupil diameter left [mm] - Tobii"] - df["Pupil diameter right [mm] - Tobii"]
        df["Pupil Difference [mm]"] = pupil_difference

        return df
    
class GGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        if model_type.feature_set == "Implicit":
            # output warning message for implicit vs. explicit models
            warnings.warn("G cannot be used as a feature in implicit models. Feature removed.")

            return []
        
        return ["magnitude - Centrifuge"]

    def process(self, df, file_paths):
        # Process magnitude Centrifuge column to include 1.2g instead of NaN
        df.fillna({"magnitude - Centrifuge": 1.2}, inplace = True)

        return df
    
class CognitiveGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        return ["deviation - Cog", "RespTime - Cog", "Correct - Cog", "joystickPosMag - Cog", "joystickVelMag - Cog"]
    
    def process(self, df, file_paths):
        # Adjust columns of data frame for feature
        df["Correct - Cog"].replace({
            "correct": 1,
            "no response": 0,
            "incorrect": -1,
            "NO VALUE": np.nan
        }, inplace = True)

        # Output warning message for fnirs
        warnings.warn("Per information from Chris on 03/12/25, Cognitive data only collected right before and right after ROR.")
        # Note post 3/12 meeting: the target is stationary, while the participant moves the tracker
        # so this metric no longer makes sense

        return df

class RawEEGGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        raw_eeg_features = self.get_separated_feature_names()

        # Pull condition-specific EEG streams
        if model_type.afe_filter == "Complete":
            return raw_eeg_features["Shared"] + raw_eeg_features["AFE Only"]
        elif model_type.afe_filter == "noAFE":
            return raw_eeg_features["Shared"] + raw_eeg_features["Non-AFE Only"]
        else: # Use shared features only
            return raw_eeg_features["Shared"]

    @staticmethod
    def get_separated_feature_names():
        """
        Returns a dictionary with separate lists of shared, AFE-only, and non-AFE EEG features.

        Parameters:
            None

        Returns:
            dict: A dictionary with keys "Shared", "AFE Only", and "Non-AFE Only", each containing a list of corresponding EEG feature names.
        """

        return {
            "Shared": [
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
                "P8 - EEG"
            ],
            "AFE Only": [
                "F4 - EEG", 
                "T7 - EEG", 
                "O1 - EEG", 
                "O2 - EEG"
            ],
            "Non-AFE Only": [
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
                "P6 - EEG"
            ]
        }

class ProcessedEEGGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        processed_eeg_features = self.get_separated_feature_names()

        # Pull condition-specific EEG streams
        if model_type.afe_filter == "Complete":
            return processed_eeg_features["Shared"] + processed_eeg_features["AFE Only"]
        elif model_type.afe_filter == "noAFE":
            return processed_eeg_features["Shared"] + processed_eeg_features["Non-AFE Only"]
        else: # Use shared features
            return processed_eeg_features["Shared"]
        
    @staticmethod
    def get_separated_feature_names():
        return {
            "Shared": [
                "Fz_delta - EEG", "Fz_theta - EEG", "Fz_alpha - EEG", "Fz_beta - EEG",
                "F3_delta - EEG", "F3_theta - EEG", "F3_alpha - EEG", "F3_beta - EEG",
                "C3_delta - EEG", "C3_theta - EEG", "C3_alpha - EEG", "C3_beta - EEG",
                "C4_delta - EEG", "C4_theta - EEG", "C4_alpha - EEG", "C4_beta - EEG",
                "CP1_delta - EEG", "CP1_theta - EEG", "CP1_alpha - EEG", "CP1_beta - EEG",
                "CP2_delta - EEG", "CP2_theta - EEG", "CP2_alpha - EEG", "CP2_beta - EEG",
                "T8_delta - EEG", "T8_theta - EEG", "T8_alpha - EEG", "T8_beta - EEG",
                "TP9_delta - EEG", "TP9_theta - EEG", "TP9_alpha - EEG", "TP9_beta - EEG",
                "TP10_delta - EEG", "TP10_theta - EEG", "TP10_alpha - EEG", "TP10_beta - EEG",
                "P7_delta - EEG", "P7_theta - EEG", "P7_alpha - EEG", "P7_beta - EEG",
                "P8_delta - EEG", "P8_theta - EEG", "P8_alpha - EEG", "P8_beta - EEG"
            ],
            "AFE Only": [
                "F4_delta - EEG", "F4_theta - EEG", "F4_alpha - EEG", "F4_beta - EEG",
                "T7_delta - EEG", "T7_theta - EEG", "T7_alpha - EEG", "T7_beta - EEG",
                "O1_delta - EEG", "O1_theta - EEG", "O1_alpha - EEG", "O1_beta - EEG",
                "O2_delta - EEG", "O2_theta - EEG", "O2_alpha - EEG", "O2_beta - EEG"
            ],
            "Non-AFE Only": [
                "F1_delta - EEG", "F1_theta - EEG", "F1_alpha - EEG", "F1_beta - EEG",
                "AFz_delta - EEG", "AFz_theta - EEG", "AFz_alpha - EEG", "AFz_beta - EEG",
                "AF4_delta - EEG", "AF4_theta - EEG", "AF4_alpha - EEG", "AF4_beta - EEG",
                "FT9_delta - EEG", "FT9_theta - EEG", "FT9_alpha - EEG", "FT9_beta - EEG",
                "FT10_delta - EEG", "FT10_theta - EEG", "FT10_alpha - EEG", "FT10_beta - EEG",
                "FC5_delta - EEG", "FC5_theta - EEG", "FC5_alpha - EEG", "FC5_beta - EEG",
                "FC3_delta - EEG", "FC3_theta - EEG", "FC3_alpha - EEG", "FC3_beta - EEG",
                "FC1_delta - EEG", "FC1_theta - EEG", "FC1_alpha - EEG", "FC1_beta - EEG",
                "FC2_delta - EEG", "FC2_theta - EEG", "FC2_alpha - EEG", "FC2_beta - EEG",
                "FC4_delta - EEG", "FC4_theta - EEG", "FC4_alpha - EEG", "FC4_beta - EEG",
                "FC6_delta - EEG", "FC6_theta - EEG", "FC6_alpha - EEG", "FC6_beta - EEG",
                "C5_delta - EEG", "C5_theta - EEG", "C5_alpha - EEG", "C5_beta - EEG",
                "Cz_delta - EEG", "Cz_theta - EEG", "Cz_alpha - EEG", "Cz_beta - EEG",
                "CP5_delta - EEG", "CP5_theta - EEG", "CP5_alpha - EEG", "CP5_beta - EEG",
                "CP6_delta - EEG", "CP6_theta - EEG", "CP6_alpha - EEG","CP6_beta - EEG",
                "P5_delta - EEG", "P5_theta - EEG", "P5_alpha - EEG", "P5_beta - EEG",
                "P3_delta - EEG", "P3_theta - EEG", "P3_alpha - EEG", "P3_beta - EEG",
                "P1_delta - EEG", "P1_theta - EEG", "P1_alpha - EEG", "P1_beta - EEG",
                "Pz_delta - EEG", "Pz_theta - EEG", "Pz_alpha - EEG", "Pz_beta - EEG",
                "P4_delta - EEG", "P4_theta - EEG", "P4_alpha - EEG", "P4_beta - EEG",
                "P6_delta - EEG", "P6_theta - EEG", "P6_alpha - EEG", "P6_beta - EEG"
            ]
        }

class StrainGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        if model_type.feature_set == "Implicit":
            # Output warning message for implicit vs. explicit models
            warnings.warn("Strain cannot be used as a feature in implicit models. Feature removed.")

            return []
        
        return ["Strain [0/1]"]
    
    def process(self, df, file_paths):
        # Check if using reduced dataset by looking at the filename
        main_filename = file_paths.get("main", "")
        use_reduced_dataset = "reduced" in main_filename
        
        if use_reduced_dataset:
            # Skip adding missing strain data for reduced dataset
            df, gloc_trial = df, df["trial_id"]
        else:
            # Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet
            df, gloc_trial = self.add_missing_strain(df)

        # Create Strain Vector
        event = df["event"].to_numpy()
        event_validated = df["event_validated"].to_numpy()
        strain_event = np.zeros(event.shape)

        # Find labeled 'strain' and 'end GOR' markings in the event column
        strain_indices = np.argwhere(event == "strain during GOR")
        end_GOR_indices = np.argwhere(event_validated == "end GOR")

        # Determine which trial strain label and end GOR label occur
        trial_strain = gloc_trial[strain_indices[:, 0]]
        trial_end_GOR = gloc_trial[end_GOR_indices[:, 0]]

        # when strain and eng GOR label occur on the same trial, set chunk from
        # start of strain to end of GOR to 1, otherwise 0. This was implemented because
        # some labels were missed.
        for i in range(trial_strain.shape[0]):
            if trial_end_GOR.str.contains(trial_strain.iloc[i]).any():
                trial_end_GOR_contains_strain = trial_end_GOR.str.contains(trial_strain.iloc[i])
                end_gor_trial_index = trial_end_GOR_contains_strain[trial_end_GOR_contains_strain].index.tolist()
                strain_event[strain_indices[i, 0]:end_gor_trial_index[0]] = 1

        df["Strain [0/1]"] = strain_event

        return df

    def add_missing_strain(self, gloc_data_reduced):
        ######### Add missing strain during GOR labels based on GLOC Effectiveness Spreadsheet #########

        # Add individual strain labels for trials where label was 'missed' (from column E of GLOC_Effectiveness spreadsheet)
        # Define trial & g magnitude variable
        gloc_trial = gloc_data_reduced['trial_id']
        magnitude_g = gloc_data_reduced['magnitude - Centrifuge'].to_numpy()
        event = gloc_data_reduced['event']

        ######## Trial 04-06 (GLOC_Effectiveness stain value of 6.1g) ########
        trial_individual_coding = '04-06'
        g_level_strain = 6.1

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        #
        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 06-02 (GLOC_Effectiveness stain value of 8.1g) ########
        trial_individual_coding = '06-02'
        g_level_strain = 8.1

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 06-06 End GOR Label in Event Validated ########
        trial_individual_coding = '06-06'

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        return_to_base_spin_candidates = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0]
        if len(return_to_base_spin_candidates) > 0:
            return_to_base_spin_index = return_to_base_spin_candidates[0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 07-06 (GLOC_Effectiveness stain value of 4.6g) ########
        trial_individual_coding = '07-06'
        g_level_strain = 4.6

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        ######## Trial 08-02 (GLOC_Effectiveness stain value of 8.3g) ########
        trial_individual_coding = '08-02'
        g_level_strain = 8.3

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 08-05 (GLOC_Effectiveness stain value of 4.3g) ########
        trial_individual_coding = '08-05'
        g_level_strain = 4.3

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        ## Add missing 'end GOR' label
        # Find first nan in g magnitude post GOR peak
        return_to_base_spin_candidates = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0]
        if len(return_to_base_spin_candidates) > 0:
            return_to_base_spin_index = return_to_base_spin_candidates[0]

            # Find the index of new end GOR label in full length csv
            end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index

            # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
            gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 08-06 (GLOC_Effectiveness stain value of 8.2g) ########
        trial_individual_coding = '08-06'
        g_level_strain = 8.2

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 09-03 (GLOC_Effectiveness stain value of 8.7g) ########
        trial_individual_coding = '09-03'
        g_level_strain = 8.7

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        ######## Trial 09-05 (GLOC_Effectiveness stain value of 4.8g) ########
        trial_individual_coding = '09-05'
        g_level_strain = 4.8

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 10-05 (GLOC_Effectiveness stain value of 3.8g) ########
        trial_individual_coding = '10-05'
        g_level_strain = 3.8

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        ######## Trial 10-06 (GLOC_Effectiveness stain value of 5.5g) ########
        trial_individual_coding = '10-06'
        g_level_strain = 5.5

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]
        current_event = event[trial_index]

        #### Remove original label in event column ####
        # Find row containing the string
        mislabel_mask_strain = current_event.str.contains('strain during GOR')
        # mislabel_mask_end_GOR = current_event.str.contains('end GOR')

        # Get the index of that row
        mislabel_index_strain = current_event[mislabel_mask_strain].index
        # mislabel_index_end_GOR = current_event[mislabel_mask_end_GOR].index

        # Find the index of new strain label in full length csv
        strain_relabel_index = mislabel_index_strain
        # end_GOR_relabel_index = mislabel_index_end_GOR

        # gloc_data_reduced['event'][strain_relabel_index] = None
        gloc_data_reduced.loc[strain_relabel_index, 'event'] = None
        # # gloc_data_reduced['event'][end_GOR_relabel_index] = None
        # gloc_data_reduced.loc[end_GOR_relabel_index, 'event'] = None

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 11-02 (GLOC_Effectiveness stain value of 5.5g) ########
        trial_individual_coding = '11-02'
        g_level_strain = 5.5

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 12-03 (GLOC_Effectiveness stain value of 3.7g) ########
        trial_individual_coding = '12-03'
        g_level_strain = 3.7

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        ######## Trial 13-02 (GLOC_Effectiveness stain value of 7.3g) ########
        trial_individual_coding = '13-02'
        g_level_strain = 7.3

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        ######## Trial 13-04 (GLOC_Effectiveness stain value of 7.4g) ########
        trial_individual_coding = '13-04'
        g_level_strain = 7.4

        # Find indices associated with current trial
        trial_index = (gloc_data_reduced['trial_id'] == trial_individual_coding)
        magnitude_g_trial = magnitude_g[trial_index]

        # Find index of closest magnitude value (using 5000 times steps (200s) as GOR duration)
        gor_peak = np.nanargmax(magnitude_g_trial[0:5000])
        magnitude_difference = np.abs(magnitude_g_trial - g_level_strain)[0:gor_peak]
        closest_value_strain_index = np.nanargmin(magnitude_difference)

        # Find the index of new strain label in full length csv
        strain_label_index = trial_index.idxmax() + closest_value_strain_index

        # gloc_data_reduced['event'][strain_label_index] = 'strain during GOR'
        gloc_data_reduced.loc[strain_label_index, 'event'] = 'strain during GOR'

        # ## Add missing 'end GOR' label
        # # Find first nan in g magnitude post GOR peak
        # return_to_base_spin_index = np.where(magnitude_g_trial[gor_peak:-1] == 1.2)[0][0]
        #
        # # Find the index of new end GOR label in full length csv
        # end_GOR_label_index = trial_index.idxmax() + gor_peak + return_to_base_spin_index
        #
        # # gloc_data_reduced['event'][end_GOR_label_index] = 'end GOR'
        # gloc_data_reduced.loc[end_GOR_label_index, 'event_validated'] = 'end GOR'

        return gloc_data_reduced, gloc_trial
    
class DemographicsGroup(BaseFeatureGroup):
    def get_feature_names(self, model_type):
        if model_type.feature_set == "Implicit":
            # Output warning message for implicit vs. explicit models
            warnings.warn("Demographics cannot be used as features in implicit models. Features removed.")

            return []

        return self.demographic_names

    def process(self, df, file_paths):
        # Read and process demographics data
        df, demographic_names = self.read_and_process_demographics(file_paths["demographic"], df)

        # Save for when calling get_feature_names
        self.demographic_names = demographic_names

        return df

    def read_and_process_demographics(self, demographic_data_filename, gloc_data_reduced):
        """
        This function imports the demographics spreadsheet and appends the demographics
        to the gloc_data_reduced variable. Each column represents one of the demographics
        variables below. The column is filled by determining the participant for a specific
        trial and filling the entire trial with that participants demographic data.
        """
        # Import demographics spreadsheet
        demographics = pd.read_csv(demographic_data_filename)
        demographics = demographics.astype({col: 'float32' for col in demographics.select_dtypes(include='float64').columns})
        demographics = demographics.copy()

        # Grab variables of interest
        participant_index = demographics['GLOC ID']                                                         # Corresponds to subject 1-13
        participant_gender = pd.Series(demographics['Gender code [1/0]'])        # 0 = Female, 1 = Male
        participant_age = pd.Series(demographics['Age [yr]'])
        participant_height = pd.Series(demographics['height [m]'])
        participant_weight = pd.Series(demographics['weight (kg)'])
        participant_BMI = pd.Series(demographics['BMI [kg/m^2]'])
        participant_blood_volume = pd.Series(demographics['Blood Volume [L]'])   # Based on Nadler's approximation
        # participant_PWV = pd.Series(demographics['Avg PWV'])                   # Pulse Wave Velocity
            # (no val for participant 4)
        # participant_PTT = demographics['Avg PTT [ms]'])                        # Pulse Transit Time
            # (no val for participant 4)
        participant_SBP_seated = pd.Series(demographics['Resting SBP (seat)'])   # Systolic Blood Pressure
        participant_SBP_stand = pd.Series(demographics['resting SBP (stand)'])
        participant_SBP_exercise = pd.Series(demographics['SBP after squat'])
        participant_DBP_seated = pd.Series(demographics['Resting DBP (seat)'] )  # Diastolic Blood Pressure
        participant_DBP_stand = pd.Series(demographics['resting DBP (stand)'])
        participant_DBP_exercise = pd.Series(demographics['DBP after squat'])
        participant_MAP_seated = pd.Series(demographics['Resting MAP'])          # Mean Arterial Pressure
        participant_MAP_stand = pd.Series(demographics['Resting MAP (stand'])
        participant_MAP_exercise = pd.Series(demographics['Post-Squat MAP'])
        participant_HR_seated = pd.Series(demographics['resting HR [seated]'])
        participant_HR_stand = pd.Series(demographics['resting HR (stand)'])
        participant_HR_exercise = pd.Series(demographics['HR after squat'])
        participant_max_leg_strength = pd.Series(demographics['Max (N)'])        # Max Leg Strength
        participant_largest_leg_circumference = pd.Series(demographics['largest leg circ. [cm]'])
        participant_lower_leg_volume = pd.Series(demographics['lower leg volume [mL]'])
        participant_skinfolds_chest_avg = pd.Series(demographics['chest avg'])   # Skin Folds to Approximate Body Fat %
        participant_skinfolds_abd_avg = pd.Series(demographics['abd avg'])
        participant_skinfolds_thigh_avg = pd.Series(demographics['thigh avg'])
        participant_skinfolds_midax_avg = pd.Series(demographics['midax avg'])
        participant_skinfolds_subscap_avg = pd.Series(demographics['subscap avg'])
        participant_skinfolds_tri_avg = pd.Series(demographics['tri avg'])
        participant_skinfolds_supra_avg = pd.Series(demographics['supra avg'])
        participant_skinfolds_sum = pd.Series(demographics['sum'])
        participant_percent_fat = pd.Series(demographics['% fat'])
        participant_leg_length = pd.Series(demographics['leg avg'])
        participant_arm_length = pd.Series(demographics['arm avg'])
        participant_midline_neck_length = pd.Series(demographics['neck (MNL) avg'])
        participant_lateral_neck_length = pd.Series(demographics['neck (LNL) avg'])
        participant_torso_length_post = pd.Series(demographics['torso (post) avg '])
        participant_torso_length_ax = pd.Series(demographics['torso (ax) avg '])
        participant_head_to_heart = pd.Series(demographics['head to heart avg'])
        participant_head_girth = pd.Series(demographics['head avg'])
        participant_neck_girth = pd.Series(demographics['neck avg'])
        participant_chest_upper_girth = pd.Series(demographics['chest upper avg'])
        participant_chest_under_girth = pd.Series(demographics['chest under avg'])
        participant_waist_girth = pd.Series(demographics['waist avg'])
        participant_hip_girth = pd.Series(demographics['hip avg'])
        participant_thigh_girth = pd.Series(demographics['thigh avg'])
        participant_calf_girth = pd.Series(demographics['calf avg'])
        participant_biceps_girth_flex = pd.Series(demographics['bicep flex avg'])
        participant_biceps_girth_relax = pd.Series(demographics['bicep relax avg'])
        participant_neck_flexion = pd.Series(demographics['avg (N) flexion'])
        participant_neck_extension = pd.Series(demographics['avg (N) extens'])
        participant_neck_right_rotation = pd.Series(demographics['avg (N) Rt. Rot'])
        participant_neck_left_rotation = pd.Series(demographics['avg (N) left rot'])
        participant_neck_left_lat_flex = pd.Series(demographics['avg (N) left lat flex'])
        participant_neck_right_lat_flex = pd.Series(demographics['avg (N) rt lat flex'])
        participant_pred_vo2 = pd.Series(demographics['pred. Vo2'])              # Predicted VO2

        # Concatenate all demographics of interest
        all_demographics = pd.concat([participant_gender, participant_age, participant_height, participant_weight,
                                    participant_BMI, participant_blood_volume, participant_SBP_seated,
                                    participant_SBP_stand, participant_SBP_exercise, participant_DBP_seated,
                                    participant_DBP_stand, participant_DBP_exercise, participant_MAP_seated,
                                    participant_MAP_stand, participant_MAP_exercise, participant_HR_seated,
                                    participant_HR_stand, participant_HR_exercise, participant_max_leg_strength,
                                    participant_largest_leg_circumference, participant_lower_leg_volume,
                                    participant_skinfolds_chest_avg, participant_skinfolds_abd_avg,
                                    participant_skinfolds_thigh_avg, participant_skinfolds_midax_avg,
                                    participant_skinfolds_subscap_avg, participant_skinfolds_tri_avg,
                                    participant_skinfolds_supra_avg, participant_skinfolds_sum,
                                    participant_percent_fat, participant_leg_length, participant_arm_length,
                                    participant_midline_neck_length, participant_lateral_neck_length,
                                    participant_torso_length_post, participant_torso_length_ax, participant_head_to_heart,
                                    participant_head_girth, participant_neck_girth, participant_chest_upper_girth,
                                    participant_chest_under_girth, participant_waist_girth, participant_hip_girth,
                                    participant_thigh_girth, participant_calf_girth, participant_biceps_girth_flex,
                                    participant_biceps_girth_relax, participant_neck_flexion,
                                    participant_neck_extension, participant_neck_right_rotation, participant_neck_left_rotation,
                                    participant_neck_left_lat_flex, participant_neck_right_lat_flex,
                                    participant_pred_vo2], axis=1)

        participant_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

        # Initialize variables
        participant_demographics = np.zeros((len(gloc_data_reduced), all_demographics.shape[1]))
        length_previous_participant_data = -1

        # Organize all participant data in array same length as gloc_data_reduced
        for i in range(len(participant_list)):
            length_current_participant_data = len(np.argwhere((gloc_data_reduced['subject'] == participant_list[i])))
            participant_demographics[length_previous_participant_data+1:length_previous_participant_data+1+length_current_participant_data, :] = all_demographics.iloc[i]
            length_previous_participant_data = length_current_participant_data + length_previous_participant_data

        # Create demographics data frame
        demographics_names = ['participant_gender', 'participant_age', 'participant_height', 'participant_weight',
                                    'participant_BMI', 'participant_blood_volume', 'participant_SBP_seated',
                                    'participant_SBP_stand', 'participant_SBP_exercise', 'participant_DBP_seated',
                                    'participant_DBP_stand', 'participant_DBP_exercise', 'participant_MAP_seated',
                                    'participant_MAP_stand', 'participant_MAP_exercise', 'participant_HR_seated',
                                    'participant_HR_stand', 'participant_HR_exercise', 'participant_max_leg_strength',
                                    'participant_largest_leg_circumference', 'participant_lower_leg_volume',
                                    'participant_skinfolds_chest_avg', 'participant_skinfolds_abd_avg',
                                    'participant_skinfolds_thigh_avg', 'participant_skinfolds_midax_avg',
                                    'participant_skinfolds_subscap_avg', 'participant_skinfolds_tri_avg',
                                    'participant_skinfolds_supra_avg', 'participant_skinfolds_sum',
                                    'participant_percent_fat', 'participant_leg_length', 'participant_arm_length',
                                    'participant_midline_neck_length', 'participant_lateral_neck_length',
                                    'participant_torso_length_post', 'participant_torso_length_ax', 'participant_head_to_heart',
                                    'participant_head_girth', 'participant_neck_girth', 'participant_chest_upper_girth',
                                    'participant_chest_under_girth', 'participant_waist_girth', 'participant_hip_girth',
                                    'participant_thigh_girth', 'participant_calf_girth', 'participant_biceps_girth_flex',
                                    'participant_biceps_girth_relax', 'participant_neck_flexion',
                                    'participant_neck_extension', 'participant_neck_right_rotation', 'participant_neck_left_rotation',
                                    'participant_neck_left_lat_flex', 'participant_neck_right_lat_flex',
                                    'participant_pred_vo2']
        demographics_concat = pd.DataFrame(participant_demographics, columns = demographics_names)

        # Append all demographic data to gloc data reduced
        gloc_data_reduced = pd.concat([gloc_data_reduced, demographics_concat], axis=1)
        return gloc_data_reduced, demographics_names
    
FEATURE_REGISTRY = {
    "ECG": ECGGroup(),
    "BR": BRGroup(),
    "temp": TempGroup(),
    "eyetracking": EyeTrackingGroup(),
    "AFE": AFEGroup(),
    "G": GGroup(),
    "cognitive": CognitiveGroup(),
    "rawEEG": RawEEGGroup(),
    "processedEEG": ProcessedEEGGroup(),
    "strain": StrainGroup(),
    "demographics": DemographicsGroup()
}