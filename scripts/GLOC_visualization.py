from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import os

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