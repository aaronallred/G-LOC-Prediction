import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
from feature_engine.selection import MRMR
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from feature_engine.selection import SelectByShuffling
from feature_engine.selection import SelectByTargetMeanPerformance
from feature_engine.selection import SelectBySingleFeaturePerformance

# Feature Selection
def feature_selection_lasso(x_train, x_test, y_train, all_features):
    """
    This function finds optimal lasso alpha parameter and fits a lasso model to determine
    most important features. This should only see the 'training' data.
    """
    # parameters to be tested on GridSearchCV
    params = {"alpha": np.arange(0.00001, 10, 500)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initializing the Model
    lasso = Lasso()

    # GridSearchCV with model, params and folds.
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(x_train, y_train)

    alpha_optimal = lasso_cv.best_params_['alpha']

    # calling the model with the best parameter
    lasso_optimal = Lasso(alpha=alpha_optimal)
    lasso_optimal.fit(x_train, y_train)

    # Using np.abs() to make coefficients positive.
    lasso_optimal_coef = np.abs(lasso_optimal.coef_)

    # plotting the Column Names and Importance of Columns.
    fig,ax = plt.subplots(figsize=(10,10))
    plt.bar(all_features, lasso_optimal_coef)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid()
    plt.title("Feature Selection Based on Lasso")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.15)
    plt.show()

    # Subset of the features with nonzero coefficient
    selected_features = np.array(all_features)[lasso_optimal_coef != 0]

    # Grab relevant feature columns from x_train and x_test
    feature_index = [index for index, element in enumerate(all_features) if element in selected_features]
    x_train = x_train[:, feature_index]
    x_test = x_test[:, feature_index]

    return x_train, x_test, selected_features

# def feature_selection_mrmr(x_training, y_training, all_features):
#     """
#     This function takes the training data and labels to complete mrmr.
#     """
#     x_train = pd.DataFrame(x_training, columns = all_features)
#     y_train = pd.Series(np.ravel(y_training))
#     selected_features = mrmr_classif(X=x_train, y=y_train, K=5)
#
#     return selected_features

def feature_selection_mrmr(x_train, y_train, x_test, all_features, n):
    """
    This function takes the training data and labels to complete mrmr.
    """
    # Build & Fit mRMR model
    mrmr_model = MRMR(method = "MIQ", regression = False, max_features = int(n))
    mrmr_model.fit(x_train, y_train)

    # Reduce train and test matrix
    x_train = mrmr_model.transform(x_train)
    x_test = mrmr_model.transform(x_test)

    # Use features to drop to determine features to keep
    features_to_drop = mrmr_model.features_to_drop_
    features_to_drop_index = [element[1:] for element in features_to_drop]
    features_to_drop_index = np.array([int(x) for x in features_to_drop_index])
    selected_features = [all_features[index] for index in range(len(all_features)) if index not in features_to_drop_index]

    return x_train, x_test, selected_features

def feature_selection_elastic_net(x_train, x_test, y_train, all_features):
    """
    This function finds optimal elastic net parameters and fits an elastic net model to determine
    most important features. This should only see the 'training' data.
    """

    # parameters to be tested on GridSearchCV
    params = {"alpha": np.arange(0.00001, 10, 500), "l1_ratio": np.arange(0, 1, 500)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

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

    # Subset of the features which have more than 0.001 importance.
    selected_features = np.array(all_features)[enet1_coef != 0]

    # Grab relevant feature columns from x_train and x_test
    feature_index = [index for index, element in enumerate(all_features) if element in selected_features]
    x_train = x_train[:, feature_index]
    x_test = x_test[:, feature_index]

    return x_train, x_test, selected_features

def feature_selection_ridge(x_train, x_test, y_train, all_features, n):
    """
    This function finds optimal ridge parameters and fits a ridge model to determine
    most important features. This should only see the 'training' data.
    """

    # parameters to be tested on GridSearchCV
    params = {"alpha": np.arange(0.00001, 10, 500)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

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

    # Determine threshold for top n% features
    threshold = np.percentile(ridge1_coef, 100 - n)
    selected_features = np.array(all_features)[ridge1_coef >= threshold]

    # Grab relevant feature columns from x_train and x_test
    feature_index = [index for index, element in enumerate(all_features) if element in selected_features]
    x_train = x_train[:, feature_index]
    x_test = x_test[:, feature_index]

    return x_train, x_test, selected_features

def dimensionality_reduction_PCA(x_train, x_test):
    """
    This function completes PCA for the training data.
    """
    pca = PCA(n_components=0.99, svd_solver = 'full')
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    explained_variance = pca.explained_variance_ratio_

    selected_features = [str(i) for i in range(0, np.shape(x_train_pca)[1])]

    # # Train a model on the transformed data
    # model = RandomForestClassifier()
    # model.fit(x_train_pca, y_train)
    #
    # # Make predictions
    # y_pred = model.predict(x_test_pca)
    #
    # # Evaluate the model
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    # f1 = metrics.f1_score(y_test, y_pred)
    # specificity = metrics.recall_score(y_test, y_pred, pos_label=0)

    return x_train_pca, x_test_pca, selected_features


def feature_selection_shuffle(x_train, x_test, y_train, all_features):
    sbs = SelectByShuffling(RandomForestClassifier(random_state=42),cv=2,random_state=42)
    sbs.fit_transform(x_train, y_train)

    # Reduce train and test matrix
    x_train = sbs.transform(x_train)
    x_test = sbs.transform(x_test)

    # Use features to drop to determine features to keep
    features_to_drop = sbs.features_to_drop_
    features_to_drop_index = [element[1:] for element in features_to_drop]
    features_to_drop_index = np.array([int(x) for x in features_to_drop_index])
    selected_features = [all_features[index] for index in range(len(all_features)) if index not in features_to_drop_index]

    # Example feature metrics
    # original_model_performance = sbs.initial_model_performance_
    # new_model_performance_drift = sbs.performance_drifts_
    # new_model_performance_drift_stddev = sbs.performance_drifts_std_

    # Example Plotting Code to modify
        # r = pd.concat([
        #     pd.Series(tr.performance_drifts_),
        #     pd.Series(tr.performance_drifts_std_)
        # ], axis=1
        # )
        # r.columns = ['mean', 'std']
        #
        # r['mean'].plot.bar(yerr=[r['std'], r['std']], subplots=True)
        #
        # plt.title("Performance drift elicited by shuffling a feature")
        # plt.ylabel('Mean performance drift')
        # plt.xlabel('Features')
        # plt.show()

    return x_train, x_test, selected_features

def feature_selection_performance(x_train, x_test, y_train, all_features):
    sfp = SelectBySingleFeaturePerformance(RandomForestClassifier(random_state=42),cv=2)
    sfp.fit(x_train, y_train)

    # Reduce train and test matrix
    x_train = sfp.transform(x_train)
    x_test = sfp.transform(x_test)

    # Use features to drop to determine features to keep
    features_to_drop = sfp.features_to_drop_
    features_to_drop_index = [element[1:] for element in features_to_drop]
    features_to_drop_index = np.array([int(x) for x in features_to_drop_index])
    selected_features = [all_features[index] for index in range(len(all_features)) if index not in features_to_drop_index]

    # Example feature parameters
    # sfp_feature_performance = sfp.feature_performance_
    # sfp_features_to_drop = sfp.features_to_drop_

    # Example Plotting Code to modify
        # r = pd.concat([
        #     pd.Series(sel.feature_performance_),
        #     pd.Series(sel.feature_performance_std_)
        # ], axis=1
        # )
        # r.columns = ['mean', 'std']
        #
        # r['mean'].plot.bar(yerr=[r['std'], r['std']], subplots=True)
        #
        # plt.title("Single feature model Performance")
        # plt.ylabel('R2')
        # plt.xlabel('Features')
        # plt.show()

    return x_train, x_test, selected_features


def target_mean_selection(x_train, x_test, y_train, all_features):

    # parameters to be tested on GridSearchCV
    params = {"threshold": np.arange(0.001, 1, 100)}

    # Number of Folds and adding the random state for replication
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initializing the Model
    tmp = SelectByTargetMeanPerformance()

    # GridSearchCV with model, params and folds.
    tmp_cv = GridSearchCV(tmp, param_grid=params, cv=kf)
    tmp_cv.fit(x_train, y_train)

    threshold_optimal = tmp_cv.best_params_['threshold']

    # calling the model with the best parameter
    tmp1 = SelectByTargetMeanPerformance(scoring="f1", threshold = threshold_optimal, cv=10, regression=False)
    tmp1.fit(x_train, y_train)

    # tmp = SelectByTargetMeanPerformance(scoring="f1", threshold = 0.01, cv=10, regression=False)
    # tmp.fit_transform(x_train, y_train)

    features_to_keep = tmp1.get_feature_names_out()

    # Convert feature output from selected_features_target_mean to index array and list of features
    selected_features_index = [element[1:] for element in features_to_keep]
    selected_features_index = np.array([int(x) for x in selected_features_index])

    selected_features = [all_features[index] for index in selected_features_index]

    # Grab relevant feature columns from x_train and x_test
    x_train = tmp1.transform(x_train)
    x_test = tmp1.transform(x_test)

    # Example feature code
    # tmp_variables = tmp.variables_
    # tmp_features_to_drop = tmp.features_to_drop_
    # tmp_feature_performance = tmp.feature_performance_
    # tmp_feature_performance_sdtdev = tmp.feature_performance_std_

    # Example Plotting Code to modify
    #     r = pd.concat([
    #         pd.Series(sel.feature_performance_),
    #         pd.Series(sel.feature_performance_std_)
    #     ], axis=1
    #     )
    #     r.columns = ['mean', 'std']
    #
    #     r['mean'].plot.bar(yerr=[r['std'], r['std']], subplots=True)
    #
    #     plt.title("Feature importance")
    #     plt.ylabel('ROC-AUC')
    #     plt.xlabel('Features')
    #     plt.show()

    return x_train, x_test, selected_features