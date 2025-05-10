import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import pickle
import math
import joblib
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

            # Find unique trial ids in the data being analyzed
            trial_id_in_data = gloc_data_reduced.trial_id.unique()
            # Iterate through all unique trial_id
            fig, ax = plt.subplots()
            current_index = (gloc_data_reduced['trial_id'] == '09-02')
            current_time = np.array(gloc_data_reduced['Time (s)'])
            current_g = np.array(gloc_data_reduced['magnitude - Centrifuge'])
            current_gloc = np.array(gloc)
            time = np.array(current_time[current_index])
            g_mag = np.array(current_g[current_index])
            gloc_trial = np.array(current_gloc[current_index])
            reduced_time_index = time > 0
            time_reduced = time[reduced_time_index]
            g_mag_reduced = g_mag[reduced_time_index]
            gloc_reduced = gloc_trial[reduced_time_index]
            ax.plot(time_reduced, gloc_reduced, label='g-loc')
            ax.plot(time_reduced, g_mag_reduced, label='centrifuge g')
            plt.xlabel('Time (s)')
            plt.ylabel('g')
            plt.title('Trial 09-02')
            plt.legend()
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
    plt.pause(1)

def plot_all_features():

    windows = np.arange(len(x_feature_matrix))
    for i in range(np.size(all_features)):
        fig, ax = plt.subplots()
        ax.plot(windows, x_feature_matrix[:,i], label=all_features[i])
        ax.autoscale()
        plt.xlabel('Window')
        plt.ylabel('Feature')
        plt.title(all_features[i])
        plt.show()

    for col in range(np.size(sliding_window_mean_current, axis = 1)):
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,number_windows_current, num=number_windows_current), sliding_window_mean_current[:,col])
        plt.xlabel('Time (s)')
        plt.ylabel('Feature')
        plt.title(combined_baseline_names[col])
        plt.show()

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

def plot_cross_val(data_dict):
    # List of model names
    model_names = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F']

    # Combine data
    all_data = []

    for label, df in data_dict.items():
        temp_df = df.copy()

        # Use the index as a column for the model name
        temp_df['model'] = temp_df.index
        temp_df['label'] = label

        all_data.append(temp_df)

    combined_df = pd.concat(all_data)

    # Metrics to plot (you can select one or loop through them)
    metrics = [col for col in combined_df.columns if col not in ['model', 'label']]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='model', y=metric, data=combined_df, inner='quartile', hue='model',palette='viridis', alpha=0.8)
        # Scatter points overlaid (stripplot with jitter)
        sns.stripplot(x='model', y=metric, data=combined_df, color='black', size=4, jitter=True)
        plt.title(f"{metric.capitalize()} Distribution Across Runs for Each Model")
        plt.xlabel("Model")
        plt.ylabel(metric.capitalize())
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)

    plt.show()

def plot_cross_val_sp(data_dict):
    # Combine all label data into one DataFrame
    all_data = []
    for label, df in data_dict.items():
        temp_df = df.copy()
        temp_df['model'] = temp_df.index
        temp_df['label'] = label
        all_data.append(temp_df)

    combined_df = pd.concat(all_data)

    # Get all metric columns (exclude non-metrics)
    metrics = [col for col in combined_df.columns if col not in ['model', 'label']]
    num_metrics = len(metrics)

    # Subplot layout (auto-fit to rows of 2)
    ncols = 2
    nrows = math.ceil(num_metrics / ncols)

    # --- Compute F1-score statistics ---
    if 'f1-score' in combined_df.columns and 'model' in combined_df.columns:
        # Group by model and calculate median, count, std, and 95% CI
        f1_stats = (
            combined_df
            .groupby('model')['f1-score']
            .agg(['median', 'count', 'std'])
            .dropna()
        )
        f1_stats['ci95'] = 1.96 * (f1_stats['std'] / (f1_stats['count'] ** 0.5))

        # Find the best model based on median F1-score
        best_model = f1_stats['median'].idxmax()
        best_median = f1_stats.loc[best_model, 'median']
        ci = f1_stats.loc[best_model, 'ci95']

        print(f"Best model by median F1-score: {best_model}")
        print(f"Median F1-score: {best_median:.4f} (95% CI: [{best_median - ci:.4f}, {best_median + ci:.4f}])")
    else:
        print("Warning: Required columns ('model', 'f1-score') not found. Skipping F1-score summary.")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows))
    axes = axes.flatten() if num_metrics > 1 else [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Violin plot with cubehelix palette
        sns.violinplot(
            x='model', y=metric, data=combined_df,
            inner='quartile', palette='viridis', hue='model',
            linewidth=1, alpha=0.8, ax=ax
        )

        # Overlay scatter points
        sns.stripplot(
            x='model', y=metric, data=combined_df,
            color='black', size=4, jitter=True, ax=ax
        )

        ax.set_title(f"{metric.capitalize()} Distribution Across Runs for Each Model ")
        ax.set_xlabel("")
        ax.set_ylabel(metric.capitalize())
        ax.tick_params(axis='x', rotation=45)

    # Hide any extra subplots if metrics < total axes
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_cross_val_hist(data_dict):

    # Combine data from all labels
    all_data = []

    for label, df in data_dict.items():
        temp_df = df.copy()

        # Use the index as a column for the model name
        temp_df['model'] = temp_df.index
        temp_df['label'] = label

        all_data.append(temp_df)

    # Combine all DataFrames
    combined_df = pd.concat(all_data)

    # Select metrics (excluding 'model' and 'label')
    metrics = [col for col in combined_df.columns if col not in ['model', 'label']]
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Overlay the histogram on top of the violin plot
        sns.histplot(data=combined_df, x=metric, hue='model', kde=True, multiple="stack", palette='colorblind', alpha=0.3,
                     bins=15, ax=ax)

        # Add titles and labels
        plt.title(f"{metric.capitalize()} Distribution Across Labels for Each Model")
        plt.xlabel("Model")
        plt.ylabel(metric.capitalize())

        # Tight layout to adjust spacing
        plt.tight_layout()

        # Show plot
        plt.show(block=False)
        plt.pause(1)
    plt.show()


def plot_f1_bar_with_ci(data_dict):
    # Combine all data into a single DataFrame
    all_data = []
    for label, df in data_dict.items():
        temp_df = df.copy()
        temp_df['model'] = temp_df.index
        temp_df['label'] = label
        all_data.append(temp_df)

    combined_df = pd.concat(all_data).reset_index(drop=True)

    # Check that required columns exist
    if 'f1-score' not in combined_df.columns or 'model' not in combined_df.columns:
        print("Error: 'f1-score' or 'model' column not found.")
        return

    # Group by model and calculate median, count, std, and 95% CI
    summary = (
        combined_df
        .groupby('model')['f1-score']
        .agg(['median', 'count', 'std'])
        .dropna()
        .rename(columns={'median': 'f1_median'})
    )
    summary['ci95'] = 1.96 * (summary['std'] / np.sqrt(summary['count']))

    # --- Print the results for all models ---
    print("F1-score summary for all models:")
    print(summary[['f1_median', 'ci95']])
    print("\n")

    # Plot the barplot first
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        x=summary.index,
        y=summary['f1_median'],
        hue=summary.index,
        palette='viridis'
    )

    # Add error bars (95% CI) on top of the bars
    for i, model in enumerate(summary.index):
        plt.errorbar(
            x=i,
            y=summary.loc[model, 'f1_median'],
            yerr=summary.loc[model, 'ci95'],
            fmt='none',
            color='black',
            capsize=5,
            elinewidth=2
        )

    plt.title('Median F1-score with 95% CI per Model')
    plt.ylabel('F1-score')
    plt.xlabel('Model')
    plt.ylim(0, 1.05)  # Ensure the y-axis starts from 0 and goes slightly above 1
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def bridget_recall_specificity_balanced_accuracy(data_dict):
    # Combine all data into a single DataFrame
    all_data = []
    for label, df in data_dict.items():
        temp_df = df.copy()
        temp_df['model'] = temp_df.index
        temp_df['label'] = label
        all_data.append(temp_df)

    combined_df = pd.concat(all_data).reset_index(drop=True)

    # Check that required columns exist
    required_columns = ['f1-score', 'recall', 'specificity', 'model']
    if not all(col in combined_df.columns for col in required_columns):
        print(f"Error: Required columns {required_columns} not found.")
        return

    # Group by model and calculate mean, count, std for f1-score, recall, and specificity
    summary = (
        combined_df
        .groupby('model')[['f1-score', 'recall', 'specificity']]
        .agg(['mean', 'count', 'std'])
        .dropna()
    )

    # Rename columns for clarity, joining the multi-level column names
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]

    # Calculate CI95 for each metric (f1, recall, specificity)
    for metric in ['f1-score', 'recall', 'specificity']:
        summary[f'{metric}_ci95'] = 1.96 * (summary[f'{metric}_std'] / np.sqrt(summary[f'{metric}_count']))

    # Calculate Balanced Accuracy: (Recall + Specificity) / 2
    summary['balanced_accuracy'] = (summary['recall_mean'] + summary['specificity_mean']) / 2
    summary['balanced_accuracy_ci95'] = 1.96 * (
                summary['recall_std'] / np.sqrt(summary['recall_count']) + summary['specificity_std'] / np.sqrt(
            summary['specificity_count']))

    # --- Print the results for all models ---
    # Set pandas display options to show all rows and columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Auto-adjust width to avoid line truncation
    pd.set_option('display.max_colwidth', None)  # Ensure no column width truncation

    print("Summary for all models:")
    print(summary[['f1-score_mean', 'f1-score_std', 'f1-score_ci95',
                   'recall_mean', 'recall_std', 'recall_ci95',
                   'specificity_mean', 'specificity_std', 'specificity_ci95',
                   'balanced_accuracy', 'balanced_accuracy_ci95']])
    print("\n")

    # Plot the barplot for each metric (F1-score, Recall, Specificity, Balanced Accuracy)
    metrics = ['f1-score', 'recall', 'specificity', 'balanced_accuracy']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(22, 6))

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # For each metric, use the proper column for y
        if metric != 'balanced_accuracy':
            sns.barplot(
                x=summary.index,
                y=summary[f'{metric}_mean'],
                hue=summary.index,  # Use the model as hue
                palette='viridis',
                ax=ax,
                legend=False  # Turn off the legend
            )
        else:
            sns.barplot(
                x=summary.index,
                y=summary['balanced_accuracy'],
                hue=summary.index,  # Use the model as hue
                palette='viridis',
                ax=ax,
                legend=False  # Turn off the legend
            )

        # Add error bars (95% CI) on top of the bars
        for j, model in enumerate(summary.index):
            if metric != 'balanced_accuracy':
                ax.errorbar(
                    x=j,
                    y=summary.loc[model, f'{metric}_mean'],
                    yerr=summary.loc[model, f'{metric}_ci95'],
                    fmt='none',
                    color='black',
                    capsize=5,
                    elinewidth=2
                )
            else:
                ax.errorbar(
                    x=j,
                    y=summary.loc[model, 'balanced_accuracy'],
                    yerr=summary.loc[model, 'balanced_accuracy_ci95'],
                    fmt='none',
                    color='black',
                    capsize=5,
                    elinewidth=2
                )

        ax.set_title(f'Mean {metric.capitalize()} with 95% CI per Model')
        ax.set_ylabel(f'{metric.capitalize()}')
        ax.set_xlabel('Model')
        ax.set_ylim(0, 1.05)  # Ensure the y-axis starts from 0 and goes slightly above 1
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_bayes_tuning(clf):
    results = clf.cv_results_

    # Extract mean test scores and per-fold scores
    mean_test_scores = results['mean_test_score']
    cv_scores = results['split0_test_score']  # example: you can access the scores of each fold (split)

    # Get best score so far at each step
    best_so_far = np.maximum.accumulate(mean_test_scores)

    # Number of iterations
    n_iter = len(mean_test_scores)

    # Plot best scores so far
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)  # Left plot: Best scores so far
    plt.plot(best_so_far, marker='o', linestyle='-', color='b')
    plt.title("BayesSearchCV Best Score Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best F1 Score so far")
    plt.grid(True)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Plot Flags
    plot_data = 0       # flag to set whether plots should be generated (0 = no, 1 = yes)
    plot_pairwise = 0   # flag to set whether pairwise plots should be generated (0 = no, 1 = yes)
    plot_cv = 1
    plot_bayes = 0

    # Visualization of feature throughout trial
    if plot_data == 1:
        initial_visualization(gloc_data_reduced, gloc, feature_baseline, all_features, time_variable)

    # Visualize sliding window mean
    if plot_data == 1:
        sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features, gloc_data_reduced)

    # Visualization of pairwise features
    if plot_pairwise == 1:
        pairwise_visualization(gloc_window, sliding_window_mean, all_features, gloc_data_reduced)

    # Visualization of k-fold cross validation
    if plot_cv == 1:

        with open('../PerformanceSave/CrossValidation/ExplicitV0-8HPO_SMOTE_LASSO_noNAN_all/CrossValidation.pkl', 'rb') as f:
            data_dict = pickle.load(f)

        plot_cross_val_sp(data_dict)
        plot_f1_bar_with_ci(data_dict)
        bridget_recall_specificity_balanced_accuracy(data_dict)
        #plot_cross_val_hist(data_dict)

    # Visualization of bayes search convergence
    if plot_bayes == 1:

        with open('../ModelSave/CV/ImplicitV0-8HPOnoFSnoNANrf2/2/random_forest_model.pkl', 'rb') as f:
            clf = joblib.load(f)

        plot_bayes_tuning(clf)

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

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(7, 14), sharex=True)

    # Top subplot: Actual vs. Predicted Labels
    axes[0].plot(predicted, label='Predicted Labels', color='red', linestyle='--', linewidth=1.5)
    axes[0].plot(ground_truth, label='Actual Labels', color='green', linewidth=1.5)
    axes[0].set_title('Actual vs. Predicted Labels')
    axes[0].set_ylabel('Label Value')
    axes[0].legend()
    axes[0].grid(True)

    # Bottom subplot: Predictors over Time
    for feature_idx in range(predictors_over_time.shape[1]):  # Loop over each feature
        axes[1].plot(predictors_over_time[:, feature_idx], label=f'Feature {feature_idx + 1}')

    axes[1].set_title('Predictors from Test Dataset Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Feature Value')
    # axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)