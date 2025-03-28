import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold
from feature_engine.selection import MRMR
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from feature_engine.selection import SelectByShuffling
from feature_engine.selection import SelectByTargetMeanPerformance
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from GLOC_classifier import *


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

    # Use optimal alpha value from grid search CV
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

    # Use optimal alpha value and l1 ratio from grid search CV
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

    # Use optimal alpha value from grid search CV
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
    This function completes PCA for the training data. The number of components are determined
    based on achieving n_components*100 percent explained variance.
    """
    # Complete PCA
    pca = PCA(n_components=0.99, svd_solver = 'full')
    pca.fit(x_train)

    # Transform training & test matrices
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Evaluate explained variance
    explained_variance = pca.explained_variance_ratio_

    # Create string name for PCA features
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


def feature_selection_shuffle(x_train, x_test, y_train, all_features, classifier_method):
    # Complete feature selection by shuffling for each classifier
    if classifier_method == 'logreg':
        sbs = SelectByShuffling(LogisticRegression(random_state=42),cv=5,random_state=42)
    elif classifier_method == 'rf':
        sbs = SelectByShuffling(RandomForestClassifier(random_state=42), cv=5, random_state=42)
    elif classifier_method == 'lda':
        sbs = SelectByShuffling(LinearDiscriminantAnalysis(), cv=5, random_state=42)
    elif classifier_method == 'knn':
        sbs = SelectByShuffling(KNeighborsClassifier(), cv=5, random_state=42)
    elif classifier_method == 'svm':
        sbs = SelectByShuffling(svm.SVC(random_state=42), cv=5, random_state=42)
    elif classifier_method == 'gb':
        sbs = SelectByShuffling(GradientBoostingClassifier(random_state=42), cv=5, random_state=42)

    # fit select by shuffling on the training data
    sbs.fit(x_train, y_train)

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

def feature_selection_performance(x_train, x_test, y_train, all_features, classifier_method):
    # Complete feature selection by single feature performance for each classifier
    if classifier_method == 'logreg':
        sfp = SelectBySingleFeaturePerformance(LogisticRegression(random_state=42),cv=5)
    elif classifier_method == 'rf':
        sfp = SelectBySingleFeaturePerformance(RandomForestClassifier(random_state=42), cv=5)
    elif classifier_method == 'lda':
        sfp = SelectBySingleFeaturePerformance(LinearDiscriminantAnalysis(), cv=5)
    elif classifier_method == 'knn':
        sfp = SelectBySingleFeaturePerformance(KNeighborsClassifier(), cv=5)
    elif classifier_method == 'svm':
        sfp = SelectBySingleFeaturePerformance(svm.SVC(random_state=42), cv=5)
    elif classifier_method == 'gb':
        sfp = SelectBySingleFeaturePerformance(GradientBoostingClassifier(random_state=42), cv=5)

    # fit single feature performance model on the training data
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

    # Use optimal value for threshold found in GridSearchCV
    threshold_optimal = tmp_cv.best_params_['threshold']

    # calling the model with the best parameter
    tmp1 = SelectByTargetMeanPerformance(scoring="f1", threshold = threshold_optimal, cv=10, regression=False)
    tmp1.fit(x_train, y_train)

    # tmp = SelectByTargetMeanPerformance(scoring="f1", threshold = 0.01, cv=10, regression=False)
    # tmp.fit_transform(x_train, y_train)

    # find features to keep
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


def ridge_methods(x_train, x_test, y_train, y_test, all_features, classifier_type, train_class, class_weight_imb):
    # Initialize Dictionaries
    performance_metric_summary_ridge = dict()
    x_train_ridge = dict()
    x_test_ridge = dict()
    selected_features_ridge = dict()

    # Set threshold range
    percentile_threshold = np.linspace(10, 100, 10)

    # Loop through threshold values
    for n in range(len(percentile_threshold)):
        # Determine reduced feature set & transform x_train and x_test
        x_train_ridge[n], x_test_ridge[n], selected_features_ridge[n] = feature_selection_ridge(x_train, x_test, y_train,
                                                                                       all_features,
                                                                                       percentile_threshold[n])

        # Assess performance for all classifiers
        performance_metric_summary_ridge[percentile_threshold[n]] = (call_all_classifiers(classifier_type, x_train_ridge[n], x_test_ridge[n], y_train,
                                                                 y_test, selected_features_ridge[n], train_class,
                                                                 class_weight_imb))

    return x_train_ridge, x_test_ridge, selected_features_ridge, performance_metric_summary_ridge


def mrmr_methods(x_train, x_test, y_train, y_test, all_features, classifier_type, train_class,
                                   class_weight_imb):
    # Initialize Dictionaries
    performance_metric_summary_mrmr = dict()
    x_train_mrmr = dict()
    x_test_mrmr = dict()
    selected_features_mrmr = dict()

    # Set number of features range
    number_features = np.linspace(round(0.1 * len(all_features)), len(all_features), 10)

    # Loop through n features
    for n in range(len(number_features)):
        # Determine reduced feature set & transform x_train and x_test
        x_train_mrmr[n], x_test_mrmr[n], selected_features_mrmr[n] = feature_selection_mrmr(x_train, y_train, x_test,
                                                                                            all_features,
                                                                                            number_features[n])

        # Assess performance for all classifiers
        performance_metric_summary_mrmr[n] = (call_all_classifiers(classifier_type, x_train_mrmr[n], x_test_mrmr[n], y_train,
                                                                   y_test, selected_features_mrmr[n], train_class,
                                                                   class_weight_imb))

    return x_train_mrmr, x_test_mrmr, selected_features_mrmr, performance_metric_summary_mrmr

def sfp_methods(x_train, x_test, y_train, y_test, all_features, train_class, class_weight_imb):

    # Initialize Dictionaries
    x_train_performance = dict()
    x_test_performance = dict()
    selected_features_performance = dict()

    # Define Classifier Methods
    classifier_method = ['logreg', 'rf', 'lda', 'knn', 'svm', 'gb']

    for i in range(len(classifier_method)):
        # Determine reduced feature set & transform x_train and x_test
        x_train_performance[i], x_test_performance[i], selected_features_performance[i] = feature_selection_performance(x_train,
                                                                                                               x_test,
                                                                                                               y_train,
                                                                                                               all_features,
                                                                                                               classifier_method[
                                                                                                                   i])

        ## Assess performance for all classifiers
        # Logistic Regression | logreg
        if classifier_method[i] == 'logreg':
            accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
                classify_logistic_regression(x_train_performance[i], x_test_performance[i], y_train, y_test, class_weight_imb,
                                             retrain=train_class))

        # Random Forrest | rf
        elif classifier_method[i] == 'rf':
            accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
                classify_random_forest(x_train_performance[i], x_test_performance[i], y_train, y_test, class_weight_imb,
                                       retrain=train_class))

        # Linear discriminant analysis | LDA
        elif classifier_method[i] == 'lda':
            accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
                classify_lda(x_train_performance[i], x_test_performance[i], y_train, y_test, retrain=train_class))

        # K Nearest Neighbors | KNN
        elif classifier_method[i] == 'knn':
            accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
                classify_knn(x_train_performance[i], x_test_performance[i], y_train, y_test, retrain=train_class))

        # Support Vector Machine | SVM
        elif classifier_method[i] == 'svm':
            accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
                classify_svm(x_train_performance[i], x_test_performance[i], y_train, y_test, class_weight_imb,
                             retrain=train_class))

        # Ensemble with Gradient Boosting | EGB
        elif classifier_method[i] == 'gb':
            accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
                classify_ensemble_with_gradboost(x_train_performance[i], x_test_performance[i], y_train, y_test,
                                                 retrain=train_class))

    # Combine performance metrics
    performance_metric_summary_sfp = (summarize_performance_metrics(accuracy_logreg, accuracy_rf, accuracy_lda,
                                                                    accuracy_knn, accuracy_svm, accuracy_gb,
                                                                    precision_logreg, precision_rf, precision_lda,
                                                                    precision_knn, precision_svm, precision_gb,
                                                                    recall_logreg, recall_rf, recall_lda, recall_knn,
                                                                    recall_svm, recall_gb, f1_logreg, f1_rf, f1_lda,
                                                                    f1_knn, f1_svm, f1_gb, specificity_logreg,
                                                                    specificity_rf, specificity_lda, specificity_knn,
                                                                    specificity_svm, specificity_gb, g_mean_logreg,
                                                                    g_mean_rf, g_mean_lda, g_mean_knn,
                                                                    g_mean_svm, g_mean_gb))

    return x_train_performance, x_test_performance, selected_features_performance, performance_metric_summary_sfp

def shuffle_methods(x_train, x_test, y_train, y_test, all_features, train_class, class_weight_imb):
    # Initialize Dictionaries
    x_train_shuffle = dict()
    x_test_shuffle = dict()
    selected_features_shuffle = dict()

    # Define Classifier Methods
    classifier_method = ['logreg', 'rf', 'svm', 'gb']

    # Loop through classifier methods
    for i in range(len(classifier_method)):
        # Determine reduced feature set & transform x_train and x_test
        x_train_shuffle[i], x_test_shuffle[i], selected_features_shuffle[i] = feature_selection_shuffle(x_train, x_test,
                                                                                                        y_train,
                                                                                                        all_features,
                                                                                                        classifier_method[
                                                                                                            i])

        ## Assess performance for all classifiers
        # Logistic Regression | logreg
        if classifier_method[i] == 'logreg':
            accuracy_logreg, precision_logreg, recall_logreg, f1_logreg, specificity_logreg, g_mean_logreg = (
                classify_logistic_regression(x_train_shuffle[i], x_test_shuffle[i], y_train, y_test, class_weight_imb,
                                             retrain=train_class))

        # Random Forrest | rf
        elif classifier_method[i] == 'rf':
            accuracy_rf, precision_rf, recall_rf, f1_rf, tree_depth, specificity_rf, g_mean_rf = (
                classify_random_forest(x_train_shuffle[i], x_test_shuffle[i], y_train, y_test, class_weight_imb,
                                       retrain=train_class))

        # Linear discriminant analysis | LDA
        elif classifier_method[i] == 'lda':
            accuracy_lda, precision_lda, recall_lda, f1_lda, specificity_lda, g_mean_lda = (
                classify_lda(x_train_shuffle[i], x_test_shuffle[i], y_train, y_test, retrain=train_class))

        # K Nearest Neighbors | KNN
        elif classifier_method[i] == 'knn':
            accuracy_knn, precision_knn, recall_knn, f1_knn, specificity_knn, g_mean_knn = (
                classify_knn(x_train_shuffle[i], x_test_shuffle[i], y_train, y_test, retrain=train_class))

        # Support Vector Machine | SVM
        elif classifier_method[i] == 'svm':
            accuracy_svm, precision_svm, recall_svm, f1_svm, specificity_svm, g_mean_svm = (
                classify_svm(x_train_shuffle[i], x_test_shuffle[i], y_train, y_test, class_weight_imb,
                             retrain=train_class))

        # Ensemble with Gradient Boosting | EGB
        elif classifier_method[i] == 'gb':
            accuracy_gb, precision_gb, recall_gb, f1_gb, specificity_gb, g_mean_gb = (
                classify_ensemble_with_gradboost(x_train_shuffle[i], x_test_shuffle[i], y_train, y_test,
                                                 retrain=train_class))

    # Define classifiers being used and summary performance meetrics to use
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1-score', 'specificity', 'g mean']

    # For each performance metric, combine each machine learning method into np array
    accuracy = np.array([accuracy_logreg, accuracy_rf, accuracy_svm, accuracy_gb])
    precision = np.array([precision_logreg, precision_rf, precision_svm, precision_gb])
    recall = np.array([recall_logreg, recall_rf, recall_svm, recall_gb])
    f1 = np.array([f1_logreg, f1_rf, f1_svm, f1_gb])
    specificity = np.array([specificity_logreg, specificity_rf, specificity_svm, specificity_gb])
    g_mean = np.array([g_mean_logreg, g_mean_rf, g_mean_svm, g_mean_gb])

    # create combined stack of all performance metrics
    combined_metrics = np.column_stack((accuracy, precision, recall, f1, specificity, g_mean))

    # label combined metrics by classifier name and performance metric name
    performance_metric_summary_shuffle = pd.DataFrame(combined_metrics, index=classifier_method,
                                                      columns=performance_metrics)

    return x_train_shuffle, x_test_shuffle, selected_features_shuffle, performance_metric_summary_shuffle