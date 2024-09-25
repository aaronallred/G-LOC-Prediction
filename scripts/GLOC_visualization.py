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
    plt.show(block=False)

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


