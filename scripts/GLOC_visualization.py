from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

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

if __name__ == "__main__":

    # Plot Flags
    plot_data = 0       # flag to set whether plots should be generated (0 = no, 1 = yes)
    plot_pairwise = 0   # flag to set whether pairwise plots should be generated (0 = no, 1 = yes)

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable)

    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced)