###### File to update a y input to ML model with the intent of predicting GLOC ######

import numpy as np

def y_prediction_including(y, backstep):
    """
    Adds predictive GLOC flags to the y array by backstepping from the first GLOC event.
    Only adds 1s, does not remove existing ones or perform any sort of shift
    """
    y = np.array(y) # Ensure correct format
    non_zero_indices = np.nonzero(y)[0] # Secures the location of the non-zero (GLOC) flags

    if len(non_zero_indices) == 0:
        # No GLOC events present, return y as is
        return y

    first_index = non_zero_indices[0] # Finds smallest index (earliest time) a GLOC occurs
    y[first_index - backstep:first_index] = 1
    return y

def y_prediction_offset(y, backstep, data_rate, trial_set):
    """
    Shifts GLOC flags to the left by 'backstep' frames.
    Truncates the beginning and pads the end with zeros.
    """
    y = np.array(y)
    offset = int(backstep * data_rate) # the actual number of indices to offset.
    # if backstep is given as seconds and data rate as hz
    # the result would be something like 5 seconds back * 25hz so 125 indices shift

    # y is passed as every single subject and trial in one so we have to break out the indices.

    unique_trials = np.unique(trial_set) # finds the unique trials within the set. Gives an array of name of each unique

    for trial in unique_trials:
        # Clearing temporary variables if they exist
        trial_indices = None
        current_y = None
        gloc_indices = None
        y_shifted = None

        # Only make corrections within this trial
        trial_indices = np.nonzero(trial_set == trial) # find indices within trial set where this unique trial was
        current_y = y[trial_indices] # the range of y we are interested in (this trial set)
        gloc_indices = np.nonzero(current_y)[0] # find gloc indices within trial. These are the locations of nonzero values in array


        if len(gloc_indices) == 0:
            # No GLOC events present, return as is
            y[trial_indices] = current_y # no change

        else:
            y_shifted = current_y[offset:] # Remove the backstep from the start
            current_y = np.append(y_shifted, [0] * offset)[:len(current_y)] # add zeros to the back
            y[trial_indices] = current_y # reassign the indices of y to what has been edited

    return y

def process_NaN_temporal(y_gloc_labels, x_feature_matrix, all_features):
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

    # Identify rows with any NaNs
    row_nan_mask = np.isnan(x_feature_matrix_noNaN_cols).any(axis=1)

    # Save indices of removed rows
    removed_row_indices = np.where(row_nan_mask)[0]

    # Keep only rows without NaNs
    x_feature_matrix_noNaN = x_feature_matrix_noNaN_cols[~row_nan_mask]
    y_gloc_labels_noNaN = y_gloc_labels[~row_nan_mask]

    return y_gloc_labels_noNaN, x_feature_matrix_noNaN, all_features, removed_row_indices