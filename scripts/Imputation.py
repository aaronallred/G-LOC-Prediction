import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

def knn_impute(labels, predictors, n_neighbors=5, scale_data=False):
    """
    Imputes missing values in the predictor array using k-Nearest Neighbors (kNN).

    Parameters:
        labels (numpy.ndarray): The label array with shape (n_samples, 1).
        predictors (numpy.ndarray): The predictor array with shape (n_samples, n_features).
        n_neighbors (int): Number of neighbors to use for kNN imputation. Default is 5. Can be a hyperparameter
        scale_data (bool): Whether to scale the data before imputation.
            Default is False. We are normalizing features before the imputation code is run.

    Returns:
        imputed_labels (numpy.ndarray): The label array after imputation.
        imputed_predictors (numpy.ndarray): The predictor array after imputation.
    """

    # Combine labels and predictors
    combined_data = np.hstack((labels, predictors))

    # Optionally scale the data
    if scale_data:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
    else:
        scaled_data = combined_data

    # Create and fit the KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
    imputed_data = imputer.fit_transform(scaled_data)

    # Separate the imputed data back into labels and predictors
    imputed_labels = imputed_data[:, 0].reshape(-1, 1)
    imputed_predictors = imputed_data[:, 1:]

    # Inverse transform to get back to original scale if scaling was applied
    if scale_data:
        imputed_data_original_scale = scaler.inverse_transform(imputed_data)
        imputed_labels = imputed_data_original_scale[:, 0].reshape(-1, 1)
        imputed_predictors = imputed_data_original_scale[:, 1:]

    return imputed_labels, imputed_predictors

def knn_impute_with_smim(labels, predictors, n_neighbors=5, scale_data=False, fdr_level=0.05):
    """
    Performs kNN imputation with Selective Missing Indicator Method (SMIM).

    Parameters:
        labels (numpy.ndarray): The label array with shape (n_samples, 1).
        predictors (numpy.ndarray): The predictor array with shape (n_samples, n_features).
        n_neighbors (int): Number of neighbors to use for kNN imputation. Default is 5. Can be a hyperparameter
        scale_data (bool): Whether to scale the data before imputation. Default is False. (Given pre-scaling)
        fdr_level (float): False discovery rate level for Benjamini-Hochberg procedure. Default is 0.05.

    Returns:
        imputed_labels (numpy.ndarray): The label array after imputation.
        imputed_predictors (numpy.ndarray): The predictor array after imputation.
    """

    # Combine labels and predictors
    combined_data = np.hstack((labels, predictors))

    # Create mask for missing values
    missing_mask = np.isnan(combined_data)

    # Perform chi-square test for all features
    p_values = []
    for j in range(combined_data.shape[1]):
        contingency_table = np.array([
            [np.sum(~missing_mask[:, j]), np.sum(missing_mask[:, j])],
            [np.sum(~missing_mask[:, j] & np.any(missing_mask, axis=1)),
             np.sum(missing_mask[:, j] & np.any(missing_mask, axis=1))]
        ])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        p_values.append(p_value)

    # Perform Benjamini-Hochberg procedure
    _, p_values_corrected, _, _ = multipletests(p_values, alpha=fdr_level, method='fdr_bh')

    # Select informative features based on corrected p-values
    informative_features = [j for j, p in enumerate(p_values_corrected) if p < fdr_level]

    # Create indicator variables only for informative features
    indicators = missing_mask[:, informative_features].astype(int)

    # Combine original data with selected indicators
    Data_with_indicators = np.hstack((combined_data, indicators))

    # Scale the data if required
    if scale_data:
        scaler = StandardScaler()
        Data_scaled = scaler.fit_transform(Data_with_indicators)
    else:
        Data_scaled = Data_with_indicators

    # Perform kNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
    imputed_data = imputer.fit_transform(Data_scaled)

    # Inverse transform if scaling was applied
    if scale_data:
        imputed_data = scaler.inverse_transform(imputed_data)

    imputed_labels = imputed_data[:, 0].reshape(-1, 1)
    imputed_predictors = imputed_data[:, 1:labels.shape[1] + predictors.shape[1]]

    # Return only the original features (without indicators)
    return imputed_labels, imputed_predictors