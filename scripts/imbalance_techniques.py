from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

def resample_rus(x_train, y_train):
    # Define Model
    rus_model = RandomUnderSampler(random_state=42)

    # Resample the Training Data
    resampled_x, resampled_y = rus_model.fit_resample(x_train, y_train)

    return resampled_x, resampled_y

def resample_ros(x_train, y_train):
    # Define Model
    ros_model = RandomOverSampler(random_state=42)

    # Resample the Training Data
    resampled_x, resampled_y = ros_model.fit_resample(x_train, y_train)

    return resampled_x, resampled_y

def resample_smote(x_train, y_train):
    # Define Model
    smote_model = SMOTE(random_state=42)

    # Resample the Training Data
    resampled_x, resampled_y = smote_model.fit_resample(x_train, y_train)

    return resampled_x, resampled_y