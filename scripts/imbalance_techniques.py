from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

def resample_rus(x_train, y_train, random_state):
    # Define Model
    rus_model = RandomUnderSampler(random_state=random_state)

    # Resample the Training Data
    resampled_x, resampled_y = rus_model.fit_resample(x_train, y_train)

    return resampled_x, resampled_y

def resample_ros(x_train, y_train, random_state):
    # Define Model
    ros_model = RandomOverSampler(random_state=random_state)

    # Resample the Training Data
    resampled_x, resampled_y = ros_model.fit_resample(x_train, y_train)

    return resampled_x, resampled_y

def resample_smote(x_train, y_train, random_state):

    # # parameters to be tested on GridSearchCV
    # param = {"k_neighbors": [3, 5, 7, 9, 11, 13, 15]}
    #
    # # Number of Folds and adding the random state for replication
    # # kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    #
    # # Initializing the Model
    # smote_initialize = SMOTE(random_state=random_state)
    #
    # # GridSearchCV with model, params and 10 stratified folds.
    # smote_model = GridSearchCV(smote_initialize, param_grid=param, cv=10)

    # Define the hyperparameter search space
    search_spaces = {
        'k_neighbors': Real(3, 15)
    }

    # Initializing the Model
    smote_initialize = SMOTE(random_state=random_state)

    # Set up BayesSearchCV
    smote_cv = BayesSearchCV(
        estimator=smote_initialize,
        search_spaces=search_spaces,
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    smote_cv.fit(x_train, np.ravel(y_train))


    # Resample the Training Data
    resampled_x, resampled_y = smote_cv.fit_resample(x_train, y_train)

    return resampled_x, resampled_y

def simple_smote(x_train, y_train, random_state):

    # Initializing the Model
    smote_model = SMOTE(random_state=random_state, k_neighbors=7)

    # Resample the Training Data
    resampled_x, resampled_y = smote_model.fit_resample(x_train, y_train)

    return resampled_x, resampled_y

