from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def initial_visualization(subject_to_plot, trial_to_plot, time, gloc, feature_baseline, subject, trial, feature_to_analyze, time_variable, all_features):
    for i in range(np.size(feature_baseline, 1)):
        plot_index = (subject == subject_to_plot) & (trial == trial_to_plot)
        fig, ax = plt.subplots()
        ax.plot(time[plot_index], gloc[plot_index], label='g-loc')
        ax.plot(time[plot_index], feature_baseline[:,i], label='feature baseline')
        plt.xlabel(time_variable)
        plt.ylabel(all_features[i])
        plt.legend()
        plt.title('Base-lined Feature over Time')
        plt.show()

def sliding_window_visualization(gloc_window, sliding_window_mean, number_windows, all_features):
    for i in range(np.size(sliding_window_mean, 1)):
        fig, ax = plt.subplots()
        windows_x = np.linspace(0,number_windows, num = number_windows)
        ax.plot(windows_x, gloc_window, label='engineered label')
        ax.plot(windows_x, sliding_window_mean[:,i], label='engineered feature')
        plt.xlabel('Windows')
        plt.ylabel('Engineered Feature:' + all_features[i] + ' & Engineered Label')
        plt.legend()
        plt.title('Engineered Feature Plot')
        plt.show()

def pairwise_visualization(gloc_window, sliding_window_mean, all_features):
    # Create a pair plot to visualize how separable the data is
    dt = np.hstack((sliding_window_mean, gloc_window))
    all_features.append('gloc_class')
    dataset = pd.DataFrame(dt, columns=all_features)
    ax = sns.pairplot(dataset, hue='gloc_class', markers=["o", "s"])
    plt.suptitle("Features Pair Plot")
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.tight_layout()
    plt.show()