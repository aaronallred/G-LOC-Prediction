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
    plt.show(block=False)

def plot_HR_data():
    x = 250
    trial = 0
    fig, ax = plt.subplots()
    trial_id_in_data = gloc_data_reduced.trial_id.unique()
    current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[trial])
    current_time = np.array(gloc_data_reduced[time_variable])
    time = current_time[current_index]
    current_lead1 = np.array(gloc_data_reduced['ECG Lead 1 - Equivital'])
    lead_1 = current_lead1[current_index]

    current_lead2 = np.array(gloc_data_reduced['ECG Lead 2 - Equivital'])
    lead_2 = current_lead2[current_index]

    ax.plot(time[0:x], lead_1[0:x], label = 'Lead 1')
    ax.plot(time[0:x], lead_2[0:x], label = 'Lead 2')

    gloc_plot = gloc[current_index]
    ax.plot(time[0:x], gloc_plot[0:x], label='g-loc')
    # ax.plot(time, gloc_data_reduced['magnitude - Centrifuge'][current_index], label='centrifuge g')

    plt.xlabel(time_variable)
    plt.ylabel('ECG Lead Data')
    plt.legend()
    plt.title(f'ECG Lead Raw Data for Subject: {trial_id_in_data[trial][0:2]} & Trial: {trial_id_in_data[trial][3:]}')

    plt.show()

    fig, ax = plt.subplots()
    current_hr  = np.array(gloc_data_reduced['HR (bpm) - Equivital'])
    hr = current_hr[current_index]

    current_hr_instant = np.array(gloc_data_reduced['HR_instant - Equivital'])
    hr_instant = current_hr_instant[current_index]

    current_hr_average = np.array(gloc_data_reduced['HR_average - Equivital'])
    hr_average = current_hr_average[current_index]

    current_hr_w_average = np.array(gloc_data_reduced['HR_w_average - Equivital'])
    hr_w_average = current_hr_w_average[current_index]

    ax.plot(time[0:x], hr[0:x], label = 'HR')
    ax.plot(time[0:x], hr_instant[0:x], label='HR_instant')
    ax.plot(time[0:x], hr_average[0:x], label='HR_average')
    ax.plot(time[0:x], hr_w_average[0:x], label='HR_w_average')

    # gloc_plot = gloc[current_index]
    # ax.plot(time[0:100], gloc_plot[0:100], label='g-loc')
    # ax.plot(time, gloc_data_reduced['magnitude - Centrifuge'][current_index], label='centrifuge g')

    plt.xlabel(time_variable)
    plt.ylabel('Equivital HR Data')
    plt.title(f'HR Data for Subject: {trial_id_in_data[trial][0:2]} & Trial: {trial_id_in_data[trial][3:]}')

    RR_interval = 60000 / hr[0:x]
    hrv_sdnn = np.nanstd(RR_interval)

    successive_difference = np.diff(RR_interval)
    hrv_rmssd = np.sqrt(np.nanmean(successive_difference ** 2))

    # Compute PNN50
    count_50ms_diff = np.sum(np.abs(successive_difference) > 50)
    hrv_pnn50 = (count_50ms_diff / len(successive_difference)) * 100

    ax.plot(time[0:x], hrv_sdnn * np.ones(x), label='SDNN')
    ax.plot(time[0:x], hrv_rmssd * np.ones(x), label='RMSSD')
    ax.plot(time[0:x], hrv_pnn50 * np.ones(x), label='PNN50')
    plt.legend()

    plt.show()

    ## Compute actual RR calcs
    # Find the indices within the specified x range
    x_min = 2.75
    x_max = 2.875
    indices_in_range = np.where((time >= x_min) & (time <= x_max))
    # Extract the time values in the range
    time_range = time[indices_in_range]
    # Extract the ECG values within the range
    ECG_lead_range = lead_1[indices_in_range]
    # Find the index of the maximum peak
    max_peak_index = np.argmax(ECG_lead_range)
    time_peak = time_range[max_peak_index]
    val_peak = ECG_lead_range[max_peak_index]

def plot_PNN50():
    index_HR = baseline_names_v0.index('HR (bpm) - Equivital_v0')
    index_HR_instant = baseline_names_v0.index('HR_instant - Equivital_v0')
    index_HR_average = baseline_names_v0.index('HR_average - Equivital_v0')
    index_HR_w_average = baseline_names_v0.index('HR_w_average - Equivital_v0')

    RR_interval_HR = 60000 / feature_window_no_baseline[:, index_HR]
    RR_interval_HR_instant = 60000 / feature_window_no_baseline[:, index_HR_instant]
    RR_interval_HR_average = 60000 / feature_window_no_baseline[:, index_HR_average]
    RR_interval_HR_w_average = 60000 / feature_window_no_baseline[:, index_HR_w_average]

    successive_difference_HR = np.diff(RR_interval_HR)
    successive_difference_HR_instant = np.diff(RR_interval_HR_instant)
    successive_difference_HR_average = np.diff(RR_interval_HR_average)
    successive_difference_HR_w_average = np.diff(RR_interval_HR_w_average)

    fig, ax = plt.subplots()
    ax.plot(new_time[1:], successive_difference_HR, label='HR')
    ax.plot(new_time[1:], successive_difference_HR_instant, label='HR_instant')
    ax.plot(new_time[1:], successive_difference_HR_average, label='HR_average')
    ax.plot(new_time[1:], successive_difference_HR_w_average, label='HR_w_average')

    plt.legend()

    plt.xlabel(time_variable)
    plt.ylabel('Successive Difference between RR Interval (at 25Hz frequency)')
    plt.title(f'HR Data for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')

    plt.show()

    fig, ax = plt.subplots()
    ax.plot(new_time, RR_interval_HR, label='HR')
    ax.plot(new_time, RR_interval_HR_instant, label='HR_instant')
    ax.plot(new_time, RR_interval_HR_average, label='HR_average')
    ax.plot(new_time, RR_interval_HR_w_average, label='HR_w_average')

    plt.legend()

    plt.xlabel(time_variable)
    plt.ylabel('RR Interval (at 25Hz frequency)')
    plt.title(f'HR Data for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')

    plt.show()

    fig, ax = plt.subplots()
    ax.plot(new_time, feature_window_no_baseline[:, index_HR], label='HR')
    ax.plot(new_time, feature_window_no_baseline[:, index_HR_instant], label='HR_instant')
    ax.plot(new_time, feature_window_no_baseline[:, index_HR_average], label='HR_average')
    ax.plot(new_time, feature_window_no_baseline[:, index_HR_w_average], label='HR_w_average')

    plt.legend()

    plt.xlabel(time_variable)
    plt.ylabel('Heart Rate')
    plt.title(f'HR Data for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')

    plt.show()


    fig, ax = plt.subplots()
    ax.plot(new_time[1:], np.gradient(successive_difference_HR), label='HR')
    ax.plot(new_time[1:], np.gradient(successive_difference_HR_instant), label='HR_instant')
    ax.plot(new_time[1:], np.gradient(successive_difference_HR_average), label='HR_average')
    ax.plot(new_time[1:], np.gradient(successive_difference_HR_w_average), label='HR_w_average')

    plt.legend()

    plt.xlabel(time_variable)
    plt.ylabel('Derivative of Successive Difference')
    plt.title(f'HR Data for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')

    plt.show()

def plot_tgt_pos():
    fig, ax = plt.subplots()
    current_index = (gloc_data_reduced['trial_id'] == '01-01')
    current_time = np.array(gloc_data_reduced[time_variable])
    time = current_time[current_index]
    target_pos_x = np.array(gloc_data_reduced['tgtposX - Cog'])
    target_pos_y = np.array(gloc_data_reduced['tgtposY - Cog'])
    target_pos_x_plot = target_pos_x[current_index]
    target_pos_y_plot = target_pos_y[current_index]
    ax.plot(time, target_pos_x_plot, label='target X')
    ax.plot(time, target_pos_y_plot, label='target Y')
    plt.xlabel(time_variable)
    plt.ylabel('TgT Position')
    plt.legend()
    plt.title(f'Target Position for Trial 01-01')
    plt.show()

    fig, ax = plt.subplots()
    current_index = (gloc_data_reduced['trial_id'] == '01-01')
    current_time = np.array(gloc_data_reduced[time_variable])
    time = current_time[current_index]
    user_delta_x = np.array(gloc_data_reduced['userdeltaX - Cog'])
    user_delta_y = np.array(gloc_data_reduced['userdeltaY - Cog'])
    user_delta_x_plot = user_delta_x[current_index]
    user_delta_y_plot = user_delta_y[current_index]
    ax.plot(time, user_delta_x_plot, label='user delta X')
    ax.plot(time, user_delta_y_plot, label='user delta Y')
    plt.show()

    fig, ax = plt.subplots()
    current_index = (gloc_data_reduced['trial_id'] == '01-01')
    current_time = np.array(gloc_data_reduced[time_variable])
    time = current_time[current_index]
    target_delta_x = np.array(gloc_data_reduced['targetDeltaX - Cog'])
    target_delta_y = np.array(gloc_data_reduced['targetDeltaY - Cog'])
    target_delta_x_plot = target_delta_x[current_index]
    taraget_delta_y_plot = target_delta_y[current_index]
    ax.plot(time, target_delta_x_plot, label='target delta X')
    ax.plot(time, taraget_delta_y_plot, label='target delta Y')
    plt.show()

    fig, ax = plt.subplots()
    current_index = (gloc_data_reduced['trial_id'] == '01-01')
    current_time = np.array(gloc_data_reduced[time_variable])
    time = current_time[current_index]
    deviation = np.array(gloc_data_reduced['deviation - Cog'])
    deviation_plot = deviation[current_index]
    ax.plot(time, deviation_plot, label='deviation')
    plt.xlabel(time_variable)
    plt.ylabel('Deviation')
    plt.legend()
    plt.title(f'Deviation for Trial 01-01')
    plt.show()

def plot_strain():

    trial_id_in_data = gloc_data_reduced.trial_id.unique()

    for i in range(np.size(trial_id_in_data)):
        fig, ax = plt.subplots()
        current_index = (gloc_data_reduced['trial_id'] == trial_id_in_data[i])
        current_time = np.array(gloc_data_reduced['Time (s)'])
        time = current_time[current_index]
        current_strain = np.array(gloc_data_reduced['Strain [0/1]'])
        strain_plot = current_strain[current_index]

        ax.plot(time, strain_plot, label = 'Strain [0/1]')

        ax.plot(time, gloc_data_reduced['magnitude - Centrifuge'][current_index], label='centrifuge g')

        plt.xlabel('Time (s)')
        plt.ylabel('Strain Data')
        plt.legend()
        plt.title(f'Strain Data for Subject: {trial_id_in_data[i][0:2]} & Trial: {trial_id_in_data[i][3:]}')

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