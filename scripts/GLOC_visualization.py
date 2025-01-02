import matplotlib.font_manager as font_manager
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable):
    """
    This function makes individual plots for all trials and all features of the trials being
    analyzed. Plots include gloc label, baselined feature, and centrifuge g level.
    """

    # Find unique trial ids in the data being analyzed
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):
        # Iterate through all features
        for j in range(np.size(feature_baseline[trial_id_in_data[i]], 1)):
            fig, ax = plt.subplots()
            current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[i])
            current_time = np.array(gloc_data_reduced[time_variable])
            time = current_time[current_index]
            ax.plot(time, gloc[current_index], label='g-loc')
            ax.plot(time, feature_baseline[trial_id_in_data[i]][:,j], label='feature baseline')
            ax.plot(time, gloc_data_reduced['magnitude - Centrifuge'][current_index], label='centrifuge g')
            plt.xlabel(time_variable)
            plt.ylabel(all_features[j])
            plt.legend()
            plt.title(f'Baselined Feature over Time for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')
            plt.show()

def sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced):
    """
    This function makes individual plots for all sliding window mean engineered features. Plots
    include engineered feature over time and associated engineered g-label.
    """

    # Find unique trial ids in the data being analyzed
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):
        current_sliding_window_mean = sliding_window_mean[trial_id_in_data[i]]

        # Iterate through all features
        for j in range(np.shape(current_sliding_window_mean)[1]):
            fig, ax = plt.subplots()
            windows_x = np.linspace(0,number_windows[trial_id_in_data[i]], num = number_windows[trial_id_in_data[i]])
            ax.plot(windows_x, gloc_window[trial_id_in_data[i]], label='engineered label')
            ax.plot(windows_x, sliding_window_mean[trial_id_in_data[i]][:,j], label='engineered feature')
            plt.xlabel('Windows')
            plt.ylabel('Engineered Feature:' + all_features[j] + ' & Engineered Label')
            plt.legend()
            plt.title(f'Engineered Feature over Time for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')
            plt.show()

def pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced):
    """
     This function makes a pairwise plot for each of the features with color coordinated labels
     for non-gloc and gloc data. Current functionality works for 1 subject and 1 trial.
     """

    # Find unique trial ids in the data being analyzed
    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    # Iterate through all unique trial_id
    for i in range(np.size(trial_id_in_data)):
        dt = np.hstack((sliding_window_mean[trial_id_in_data[i]], gloc_window[trial_id_in_data[i]]))
        all_features.append('gloc_class')
        dataset = pd.DataFrame(dt, columns=all_features)
        ax = sns.pairplot(dataset, hue='gloc_class', markers=["o", "s"])
        plt.suptitle("Features Pair Plot")
        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
        plt.tight_layout()
        plt.show()

def create_confusion_matrix(y_testing, label_predictions, model_type):
    """
     This function is used to create plots of the confusion matrix for each of the Machine
     Learning classifiers.
     """

    # Create Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_testing, label_predictions)

    # Plot Confusion Matrix
    prediction_names = ['No GLOC', 'GLOC']
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(prediction_names))
    plt.xticks(tick_marks, prediction_names)
    plt.yticks(tick_marks, prediction_names)
    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.title(f'Confusion matrix: {model_type}', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
















def EF_visualization(feature,label):
    fig, ax = plt.subplots()
    iters = np.linspace(1,len(feature),len(feature))
    ax.plot(iters, label, label='engineered label')
    ax.plot(iters, feature, label='engineered feature')
    plt.xlabel('Pair ID')
    plt.ylabel('Feature / Label')
    plt.legend()
    plt.title('Engineered Feature-Label Pairs')
    plt.show()

def plot_core_predictors(gloc_data, core_predictors):
    plt.ioff()
    # Define the columns to be excluded from plotting
    grouping_column = 'trial_id'
    x_axis_column = 'Time (s)'

    # Get all columns except the grouping and x-axis column
    columns_to_plot = [col for col in core_predictors]

    # Group by 'trial id'
    grouped = gloc_data.groupby(grouping_column)

    # Loop through each group and plot every column

    for trial_id, group in grouped:
        plt.figure(figsize=(10, 6))

        # Plot each column in 'columns_to_plot' against 'Time (s)'
        for col in columns_to_plot:
            plt.plot(group[x_axis_column], group[col], label=col)

        plt.title(f'Trial ID: {trial_id}')
        plt.xlabel(x_axis_column)
        plt.ylabel('Measurements')
        plt.legend()
        plt.grid(True)
        #plt.show()

        directory = "./output/core/"  # Replace with your desired directory
        filename = str(trial_id)+".png"
        file_path = os.path.join(directory, filename)

        # Save the plot
        plt.savefig(file_path)

def plot_EEG(gloc_data,datatype):
    # Define the columns to be excluded from plotting
    grouping_column = 'trial_id'
    x_axis_column = 'Time (s)'

    if datatype == "raw":
        exclude1 = list(gloc_data.columns[0:56])
        exclude2 = list(gloc_data.columns[88:221])
        exclude3 = list(gloc_data.columns[225:])
        exclude = exclude1 + exclude2 + exclude3
    elif datatype == "delta":
        exclude1 = list(gloc_data.columns[0:93])
        exclude2 = list(gloc_data.columns[125:])
        exclude = exclude1 + exclude2
    elif datatype == "theta":
        exclude1 = list(gloc_data.columns[0:125])
        exclude2 = list(gloc_data.columns[157:])
        exclude = exclude1 + exclude2
    else:
        print("Unknown datatype.")


    # Get all columns except the grouping and x-axis column
    columns_to_plot = [col for col in gloc_data.columns if col not in [grouping_column, x_axis_column]+exclude]

    # Group by 'trial id'
    grouped = gloc_data.groupby(grouping_column)

    # Loop through each group and plot every column

    for trial_id, group in grouped:
        plt.ioff()
        plt.figure(figsize=(10, 6))

        # Plot each column in 'columns_to_plot' against 'Time (s)'
        for col in columns_to_plot:
            plt.plot(group[x_axis_column], group[col], label=col)

        plt.title(f'Trial ID: {trial_id}')
        plt.xlabel(x_axis_column)
        plt.ylabel('Measurements')
        font = font_manager.FontProperties(size=5)
        plt.legend(prop=font)
        #plt.show()

        directory = "./output/EEG/"+ datatype +"/"  # Replace with your desired directory
        filename = str(trial_id)+".png"
        file_path = os.path.join(directory, filename)

        # Save the plot
        plt.savefig(file_path)

def roc_curve_plot(all_labels, all_preds):
    # Plot the ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

def prediction_time_plot(ground_truth, predicted,predictors_over_time):
    # Plot the true values and predicted values over time
    # plt.figure(figsize=(14, 6))
    # plt.plot(ground_truth, label='Ground Truth Values', color='blue', linewidth=2)
    # plt.plot(predicted, label='Predicted Values', color='red', linestyle='--', linewidth=2)
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.title('Time Series Predictions vs True Values')
    # plt.legend()
    # plt.show()

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Top subplot: Actual vs. Predicted Labels
    axes[0].plot(ground_truth, label='Actual Labels', color='green', linewidth=1.5)
    axes[0].plot(predicted, label='Predicted Labels', color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Actual vs. Predicted Labels')
    axes[0].set_ylabel('Label Value')
    axes[0].legend()
    axes[0].grid(True)

    # Bottom subplot: Predictors over Time
    # Assuming predictors are 2D, e.g., (batch_size, time_steps, num_features)
    for feature_idx in range(predictors_over_time.shape[1]):  # Loop over each feature
        axes[1].plot(predictors_over_time[:, feature_idx], label=f'Feature {feature_idx + 1}')

    axes[1].set_title('Predictors from Test Dataset Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Feature Value')
    # axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()