import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import GridSearchCV, KFold
import faiss

def knn_impute(predictors, n_neighbors, scale_data=False):
    """
    Imputes missing values in the predictor array using k-Nearest Neighbors (kNN).

    Parameters:
        predictors (numpy.ndarray): The predictor array with shape (n_samples, n_features).
        n_neighbors (int): Number of neighbors to use for kNN imputation. Default is 5. Can be a hyperparameter
        scale_data (bool): Whether to scale the data before imputation.
            Default is False. We are normalizing features before the imputation code is run.

    Returns:
        imputed_predictors (numpy.ndarray): The predictor array after imputation.
        indicator_matrix (numpy.ndarray): The missing indicator matrix
    """

    # Optionally scale the data
    if scale_data:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(predictors)
    else:
        scaled_data = predictors

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')

    imputed_predictors = imputer.fit_transform(scaled_data)

    # Inverse transform to get back to original scale if scaling was applied
    if scale_data:
        imputed_predictors = scaler.inverse_transform(imputed_predictors)

    # Generate the indicator matrix: 1 where values were imputed, 0 otherwise
    indicator_matrix = np.isnan(predictors).astype(int)

    return imputed_predictors, indicator_matrix

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

    # get rid of columns with no mask

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

    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')

    imputed_data = imputer.fit_transform(Data_scaled)

    # Inverse transform if scaling was applied
    if scale_data:
        imputed_data = scaler.inverse_transform(imputed_data)

    imputed_labels = imputed_data[:, 0].reshape(-1, 1)
    imputed_predictors = imputed_data[:, 1:labels.shape[1] + predictors.shape[1]]

    # Return only the original features (without indicators)
    return imputed_labels, imputed_predictors

# --- FAISS-based KNN imputation ---
def fast_knn_impute(X, k=5):
    mask = np.isnan(X)
    X_imputed = X.copy()

    # Temporarily mean impute missing values
    X_temp = np.where(mask, np.nanmean(X, axis=0), X)

    # Build FAISS index
    d = X.shape[1]  # dimension
    index = faiss.IndexFlatL2(d)
    index.add(X_temp.astype(np.float32))

    # Find k nearest neighbors
    distances, indices = index.search(X_temp.astype(np.float32), k + 1)

    # Impute missing values
    for i in range(X.shape[0]):
        neighbors = indices[i, 1:]  # skip self (first neighbor)
        for j in range(X.shape[1]):
            if mask[i, j]:
                neighbor_values = X_temp[neighbors, j]
                X_imputed[i, j] = np.mean(neighbor_values)

    return X_imputed


def faster_knn_impute(X, k=5, M=32, efSearch=64):
  """
  Perform KNN imputation using FAISS HNSW index.
  Parameters:
  - X: (n_samples, n_features) matrix with missing values as np.nan
  - k: Number of neighbors for imputation
  - M: Number of neighbors in the HNSW graph (higher = more accurate, slower)
  - efSearch: Number of candidates to consider during search (higher = better recall)
  Returns:
  - X_imputed: Matrix with missing values imputed
  """
  mask = np.isnan(X)
  X_imputed = X.copy()
  # Temporarily mean impute missing values
  X_temp = np.where(mask, np.nanmean(X, axis=0), X)
  # Build FAISS index (HNSW)
  d = X.shape[1] # dimension
  index = faiss.IndexHNSWFlat(d, M)
  index.hnsw.efSearch = efSearch
  index.add(X_temp.astype(np.float32))
  # Find k nearest neighbors
  distances, indices = index.search(X_temp.astype(np.float32), k + 1)
  # Impute missing values (skip self, which is always the first neighbor)
  for i in range(X.shape[0]):
    neighbors = indices[i, 1:] # skip self
    for j in range(X.shape[1]):
      if mask[i, j]: # Only impute missing values
        neighbor_values = X_temp[neighbors, j]
        X_imputed[i, j] = np.nanmean(neighbor_values)
  return X_imputed