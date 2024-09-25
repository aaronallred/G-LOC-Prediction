from matplotlib import pyplot as plt
import numpy as np

def initial_visualization(subject_to_plot, trial_to_plot, time, gloc, feature_baseline, subject, trial, feature_to_analyze, time_variable):
    plot_index = (subject == subject_to_plot) & (trial == trial_to_plot)
    fig, ax = plt.subplots()
    ax.plot(time[plot_index], gloc[plot_index], label='g-loc')
    ax.plot(time[plot_index], feature_baseline, label='feature baseline')
    plt.xlabel(time_variable)
    plt.ylabel(feature_to_analyze)
    plt.legend()
    plt.title('Base-lined Feature over Time')
    plt.show()

def sliding_window_visualization(gloc_window, sliding_window_mean, number_windows):
    fig, ax = plt.subplots()
    windows_x = np.linspace(0,number_windows, num = number_windows)
    ax.plot(windows_x, gloc_window, label='engineered label')
    ax.plot(windows_x, sliding_window_mean, label='engineered feature')
    plt.xlabel('Windows')
    plt.ylabel('Engineered Feature/ Engineered Label')
    plt.legend()
    plt.title('Engineered Feature Plot')
    plt.show()