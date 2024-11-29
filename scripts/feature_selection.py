import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from mrmr import mrmr_classif
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

# Feature Selection
def feature_selection_lasso(x_train, y_train, all_features):
    """
    This function finds optimal lasso alpha parameter and fits a lasso model to determine
    most important features. This should only see the 'training' data.
    """
    # parameters to be tested on GridSearchCV
    params = {"alpha": np.arange(0.00001, 10, 500)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initializing the Model
    lasso = Lasso()

    # GridSearchCV with model, params and folds.
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(x_train, y_train)

    alpha_optimal = lasso_cv.best_params_['alpha']

    # calling the model with the best parameter
    lasso1 = Lasso(alpha=alpha_optimal)
    lasso1.fit(x_train, y_train)

    # Using np.abs() to make coefficients positive.
    lasso1_coef = np.abs(lasso1.coef_)

    # plotting the Column Names and Importance of Columns.
    fig,ax = plt.subplots(figsize=(10,10))
    plt.bar(all_features, lasso1_coef)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid()
    plt.title("Feature Selection Based on Lasso")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.15)
    plt.show()

    # Subsetting the features which has more than 0.001 importance.
    feature_subset = np.array(all_features)[lasso1_coef > 0.001]

    return feature_subset

def feature_selection_mrmr(x_training, y_training, all_features):

    x_train = pd.DataFrame(x_training, columns = all_features)
    y_train = pd.DataFrame(y_training)
    selected_features = mrmr_classif(X=x_train, y=y_train, K=10)

    return selected_features

def feature_selection_elastic_net(x_train, y_train, all_features):
    """
    This function finds optimal elastic net parameters and fits an elastic net model to determine
    most important features. This should only see the 'training' data.
    """

    # parameters to be tested on GridSearchCV
    params = {"alpha": np.arange(0.00001, 10, 500), "l1_ratio": np.arange(0, 1, 500)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initializing the Model
    enet = ElasticNet()

    # GridSearchCV with model, params and folds.
    enet_cv = GridSearchCV(enet, param_grid=params, cv=kf)
    enet_cv.fit(x_train, y_train)

    alpha_optimal = enet_cv.best_params_['alpha']
    l1ratio_optimal = enet_cv.best_params_['l1_ratio']

    # calling the model with the best parameter
    enet1 = ElasticNet(alpha=alpha_optimal, l1_ratio=l1ratio_optimal)
    enet1.fit(x_train, y_train)

    # Using np.abs() to make coefficients positive.
    enet1_coef = np.abs(enet1.coef_)

    # plotting the Column Names and Importance of Columns.
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.bar(all_features, enet1_coef)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid()
    plt.title("Feature Selection Based on Elastic Net")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.15)
    plt.show()

    # Subsetting the features which has more than 0.001 importance.
    selected_features = np.array(all_features)[enet1_coef > 0.001]

    return selected_features

def feature_selection_ridge(x_train, y_train, all_features):
    """
    This function finds optimal ridge parameters and fits a ridge model to determine
    most important features. This should only see the 'training' data.
    """

    # parameters to be tested on GridSearchCV
    params = {"alpha": np.arange(0.00001, 10, 500)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initializing the Model
    ridge0 = Ridge()

    # GridSearchCV with model, params and folds.
    ridge_cv = GridSearchCV(ridge0, param_grid=params, cv=kf)
    ridge_cv.fit(x_train, y_train)

    alpha_optimal = ridge_cv.best_params_['alpha']

    # calling the model with the best parameter
    ridge1 = Ridge(alpha=alpha_optimal)
    ridge1.fit(x_train, y_train)

    # Using np.abs() to make coefficients positive.
    ridge1_coef = np.abs(ridge1.coef_)
    ridge1_coef = np.ravel(ridge1_coef)

    # plotting the Column Names and Importance of Columns.
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.bar(all_features, ridge1_coef)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid()
    plt.title("Feature Selection Based on Ridge")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.7)
    plt.show()

    # Subsetting the features which has more than 0.001 importance.
    selected_features = np.array(all_features)[ridge1_coef > 0.001]

    return selected_features