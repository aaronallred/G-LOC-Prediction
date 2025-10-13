import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

if __name__ == "__main__":

    # ## CREATE EXAMPLE SWP FIGURE
    # # import Excel sheet SWP
    # figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # X = window_sizes_to_analyze
    # Y = strides_to_analyze
    # # XX, YY = np.meshgrid(X, Y)
    # #
    # # reshaped_x = XX.reshape(-1)
    # # reshaped_y = YY.reshape(-1)
    #
    # fig = plt.figure()
    #
    # # syntax for 3-D projection
    # ax = plt.axes(projection='3d')
    #
    # # Plot the surface
    # ax.plot_trisurf(X, Y, f1, cmap = 'cool')
    # # ax.plot_surface(XX, YY, f1, cmap='cool')
    #
    # # Set labels
    # ax.set_xlabel('Window Size [s]')
    # ax.set_ylabel('Stride [s]')
    # ax.set_zlabel('F1 Score')
    #
    # plt.title(f'Baseline Window = {baseline_windows_to_analyze[0]} s')
    #
    # # Show the plot
    # plt.show()
    #
    # ## Get side view from Window Size
    # fig = plt.figure()
    #
    # # syntax for 3-D projection
    # ax = plt.axes(projection='3d')
    #
    # # Plot the surface
    # ax.plot_trisurf(X, Y, f1, cmap = 'cool')
    # # ax.plot_surface(XX, YY, f1, cmap='cool')
    #
    # # Set labels
    # ax.set_xlabel('Window Size [s]')
    # ax.set_zlabel('F1 Score')
    # ax.view_init(elev=0, azim=-90)
    #
    # ax.set_yticklabels([])
    #
    # # Show the plot
    # plt.show()
    #
    # ## Get side view from Stride
    # fig = plt.figure()
    #
    # # syntax for 3-D projection
    # ax = plt.axes(projection='3d')
    #
    # # Plot the surface
    # ax.plot_trisurf(X, Y, f1, cmap = 'cool')
    # # ax.plot_surface(XX, YY, f1, cmap='cool')
    #
    # # Set labels
    # ax.set_ylabel('Stride [s]')
    # ax.set_zlabel('F1 Score')
    # ax.view_init(elev=0, azim=0)
    #
    # ax.set_xticklabels([])
    #
    # # Show the plot
    # plt.show()

    ## plot clustered bar charts for Class Imabalnce

    # Import data
    figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    df_nan = pd.read_excel(figure_data, sheet_name='ClassImbalance')

    # Sort values within each group
    df_sorted = df_nan.sort_values(by=['Classifier', 'F1'], ascending=[True, False])

    # Get unique groups and subcategories
    classifiers = df_sorted['Classifier'].unique()
    classifier_positions = [5, 3, 2, 0, 1, 4]
    bar_width = 0.11

    # Get max number of subcategories per group
    max_subcats = df_sorted.groupby('Classifier')['Method'].count().max()

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))

    subcategory_colors = {'RUS': 'paleturquoise', 'ROS': 'mediumaquamarine', 'SMOTE': 'cornflowerblue', 'CF': 'palevioletred', 'RUS+CF': 'khaki', 'ROS+CF': 'mediumpurple', 'SMOTE+CF': 'sienna', 'NONE': 'darkblue'}

    # Plot each bar
    for i, classifier in enumerate(classifiers):
        classifier_data = df_sorted[df_sorted['Classifier'] == classifier].reset_index(drop=True)
        for j, row in classifier_data.iterrows():
            x = classifier_positions[i] + (j - len(classifier_data) / 2) * bar_width
            ax.bar(x, row['F1'], width=bar_width, label=row['Method'] if i == 0 else "", color=subcategory_colors[row['Method']])

    # Aesthetics
    ax.set_xticks(classifier_positions)
    ax.set_xticklabels(classifiers)
    ax.set_ylabel("F1 Score", fontsize=16)
    ax.set_xlabel("Classifier", fontsize=16)
    ax.set_title("Class Imbalance Method Performance per Classifier", fontsize=16, pad=16)


    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order
    order = [6, 2, 4, 0, 7, 3, 5, 1]

    plt.ylim(0.75, 1.0)

    plt.legend([handles[k] for k in order], [labels[k] for k in order], title="Class Imbalance Method", title_fontsize =16, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 1])

    plt.rcParams['font.size'] = 10
    plt.tick_params(axis='both', which='major', labelsize=16)  # Sets tick label font size

    plt.show()

    ## plot clustered bar charts for imputation results

    # Import data
    figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    df_nan = pd.read_excel(figure_data, sheet_name='NAN')

    # Sort values within each group
    df_sorted = df_nan.sort_values(by=['Classifier', 'F1'], ascending=[True, False])

    # Get unique groups and subcategories
    classifiers = df_sorted['Classifier'].unique()
    classifier_positions = [5, 3, 2, 0, 1, 4]
    bar_width = 0.14

    # Get max number of subcategories per group
    max_subcats = df_sorted.groupby('Classifier')['Method'].count().max()

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))

    subcategory_colors = {'KNN, n = 3': 'darkblue', 'KNN, n = 5': 'mediumaquamarine', 'KNN, n = 7': 'cornflowerblue', 'KNN, n = 10': 'palevioletred', 'Listwise Deletion': 'khaki'}

    # Plot each bar
    for i, classifier in enumerate(classifiers):
        classifier_data = df_sorted[df_sorted['Classifier'] == classifier].reset_index(drop=True)
        for j, row in classifier_data.iterrows():
            x = classifier_positions[i] + (j - len(classifier_data) / 2) * bar_width
            ax.bar(x, row['F1'], width=bar_width, label=row['Method'] if i == 0 else "", color=subcategory_colors[row['Method']])

    # Aesthetics
    ax.set_xticks(classifier_positions)
    ax.set_xticklabels(classifiers)
    ax.set_ylabel("F1 Score", fontsize=16)
    ax.set_xlabel("Classifier", fontsize=16)
    ax.set_title("Missing Data Technique Performance per Classifier", fontsize=16, pad=16)


    # reordering the labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order
    order = [0, 1, 3, 4, 2]

    plt.ylim(0.75, 1.0)

    plt.legend([handles[k] for k in order], [labels[k] for k in order], title="Imputation Method", title_fontsize =16, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 1])

    plt.rcParams['font.size'] = 10
    plt.tick_params(axis='both', which='major', labelsize=16)  # Sets tick label font size

    plt.show()

    ##################### Create contour plots v2 #####################

    ## set contour shared values
    contour_min, contour_max = 0.5, 1.0
    levels = np.linspace(contour_min, contour_max, 50)
    axes = []

    fig = plt.figure(figsize=(4, 14), constrained_layout=True)
    gs = gridspec.GridSpec(6, 2, width_ratios=[12, 1], wspace=0.1, hspace=0.45, right=0.82)

    # log reg
    figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    df_swp = pd.read_excel(figure_data, sheet_name='SWP-logreg')

    # convert df to numpy array
    window_sizes_to_analyze = df_swp['window size'].to_numpy()
    strides_to_analyze = df_swp['stride'].to_numpy()
    baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    f1 = df_swp['f1'].to_numpy()

    x = window_sizes_to_analyze
    y = strides_to_analyze

    # Find index of max f1 score
    max_index = np.argmax(f1)

    # Determine the optimal x and y value to plot
    optimal_x = x[max_index]
    optimal_y = y[max_index]

    ax = fig.add_subplot(gs[0, 0], sharex=axes[0] if axes else None)
    axes.append(ax)

    contour = ax.tricontourf(x, y, f1, levels=levels, vmin=contour_min, vmax=contour_max, cmap="viridis", extend='min')
    # plt.colorbar(contour, label='F1 Score')
    ax.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    ax.set_title(f'Logistic Regression \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10, pad=6)
    # plt.xlabel('Window Size [s]')
    ax.set_ylabel('Stride [s]', fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])

    # RF
    df_swp = pd.read_excel(figure_data, sheet_name='SWP-rf')

    # convert df to numpy array
    window_sizes_to_analyze = df_swp['window size'].to_numpy()
    strides_to_analyze = df_swp['stride'].to_numpy()
    baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    f1 = df_swp['f1'].to_numpy()

    x = window_sizes_to_analyze
    y = strides_to_analyze

    # Find index of max f1 score
    max_index = np.argmax(f1)

    # Determine the optimal x and y value to plot
    optimal_x = x[max_index]
    optimal_y = y[max_index]

    ax = fig.add_subplot(gs[1, 0], sharex=axes[0] if axes else None)
    axes.append(ax)

    contour = ax.tricontourf(x, y, f1, levels=levels, vmin=contour_min, vmax=contour_max, cmap="viridis", extend='min')
    # plt.colorbar(contour, label='F1 Score')
    ax.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    ax.set_title(f'Random Forest \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10, pad=6)
    # plt.xlabel('Window Size [s]')
    ax.set_ylabel('Stride [s]', fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])

    # LDA
    df_swp = pd.read_excel(figure_data, sheet_name='SWP-lda')

    # convert df to numpy array
    window_sizes_to_analyze = df_swp['window size'].to_numpy()
    strides_to_analyze = df_swp['stride'].to_numpy()
    baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    f1 = df_swp['f1'].to_numpy()

    x = window_sizes_to_analyze
    y = strides_to_analyze

    # Find index of max f1 score
    max_index = np.argmax(f1)

    # Determine the optimal x and y value to plot
    optimal_x = x[max_index]
    optimal_y = y[max_index]

    ax = fig.add_subplot(gs[2, 0], sharex=axes[0] if axes else None)
    axes.append(ax)

    contour = plt.tricontourf(x, y, f1, levels=levels, vmin=contour_min, vmax=contour_max, cmap="viridis", extend='min')
    # plt.colorbar(contour, label='F1 Score')
    ax.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    ax.set_title(f'Linear Discriminant Analysis \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10, pad=6)
    # plt.xlabel('Window Size [s]')
    ax.set_ylabel('Stride [s]', fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])

    # KNN
    df_swp = pd.read_excel(figure_data, sheet_name='SWP-knn')

    # convert df to numpy array
    window_sizes_to_analyze = df_swp['window size'].to_numpy()
    strides_to_analyze = df_swp['stride'].to_numpy()
    baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    f1 = df_swp['f1'].to_numpy()

    x = window_sizes_to_analyze
    y = strides_to_analyze

    # Find index of max f1 score
    max_index = np.argmax(f1)

    # Determine the optimal x and y value to plot
    optimal_x = x[max_index]
    optimal_y = y[max_index]

    ax = fig.add_subplot(gs[3, 0], sharex=axes[0] if axes else None)
    axes.append(ax)

    contour = plt.tricontourf(x, y, f1, levels=levels, vmin=contour_min, vmax=contour_max, cmap="viridis", extend='min')
    # plt.colorbar(contour, label='F1 Score')
    ax.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    ax.set_title(f'K Nearest Neighbors \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10, pad=6)
    # plt.xlabel('Window Size [s]')
    ax.set_ylabel('Stride [s]', fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])

    # SVM
    df_swp = pd.read_excel(figure_data, sheet_name='SWP-svm')

    # convert df to numpy array
    window_sizes_to_analyze = df_swp['window size'].to_numpy()
    strides_to_analyze = df_swp['stride'].to_numpy()
    baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    f1 = df_swp['f1'].to_numpy()

    x = window_sizes_to_analyze
    y = strides_to_analyze

    # Find index of max f1 score
    max_index = np.argmax(f1)

    # Determine the optimal x and y value to plot
    optimal_x = x[max_index]
    optimal_y = y[max_index]

    ax = fig.add_subplot(gs[4, 0], sharex=axes[0] if axes else None)
    axes.append(ax)

    contour = plt.tricontourf(x, y, f1, levels=levels, vmin=contour_min, vmax=contour_max, cmap="viridis", extend='min')
    # plt.colorbar(contour, label='F1 Score')
    ax.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    ax.set_title(f'Support Vector Machine \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10, pad=6)
    # plt.xlabel('Window Size [s]')
    ax.set_ylabel('Stride [s]', fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])

    # GBC
    df_swp = pd.read_excel(figure_data, sheet_name='SWP-egb')

    # convert df to numpy array
    window_sizes_to_analyze = df_swp['window size'].to_numpy()
    strides_to_analyze = df_swp['stride'].to_numpy()
    baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    f1 = df_swp['f1'].to_numpy()

    x = window_sizes_to_analyze
    y = strides_to_analyze

    # Find index of max f1 score
    max_index = np.argmax(f1)

    # Determine the optimal x and y value to plot
    optimal_x = x[max_index]
    optimal_y = y[max_index]

    ax = fig.add_subplot(gs[5, 0], sharex=axes[0] if axes else None)
    axes.append(ax)

    contour = plt.tricontourf(x, y, f1, levels=levels, vmin=contour_min, vmax=contour_max, cmap="viridis", extend='min')
    # plt.colorbar(contour, label='F1 Score')
    ax.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    ax.set_title(f'Gradient Boosting Classifier \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10, pad=6)
    # plt.xlabel('Window Size [s]')
    ax.set_ylabel('Stride [s]', fontsize=10)
    ax.set_yticks([1, 2, 3, 4, 5])

    axes[-1].set_xlabel('Window Size [s]', fontsize=10)

    cax = fig.add_subplot(gs[:, 1])
    cbar = fig.colorbar(contour, cax=cax, ticks=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1.00])
    cbar.set_label("F1 Score", fontsize=10, labelpad=4)

    # Define a formatter function to limit to two decimal places
    formatter = mticker.FormatStrFormatter('%.2f')
    cbar.ax.yaxis.set_major_formatter(formatter)

    gs.update(top=0.96, bottom=0.04)
    plt.show()

    ##################### Create contour plots #####################

    # ## set contour shared values
    # contour_min, contour_max = 0.5, 1.0
    # levels = np.linspace(contour_min, contour_max, 10)
    # axes = []
    #
    # #log reg
    # figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-logreg')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.figure(figsize=(4, 14))
    # gs = gridspec.GridSpec(6, 2, width_ratios=[10, 1], wspace=0.05, hspace=0.5)
    # plt.subplot(6,1,1)
    # contour = plt.tricontourf(x, y, f1, cmap = "viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o',color='red', markersize=16)
    # plt.title(f'Logistic Regression \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # #plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # RF
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-rf')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 2)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o',color='red', markersize=16)
    # #plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    # plt.title(f'Random Forest \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    #
    # # LDA
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-lda')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 3)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o',color='red', markersize=16)
    # plt.title(f'Linear Discriminant Analysis \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # #plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # KNN
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-knn')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 4)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o',color='red', markersize=16)
    # plt.title(f'K Nearest Neighbors \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # #plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # SVM
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-svm')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 5)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o',color='red', markersize=16)
    # plt.title(f'Support Vector Machine \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]', fontsize=10)
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    #
    # # GBC
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-egb')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 6)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o',color='red', markersize=16)
    # plt.title(f'Gradient Boosting Classifier \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # plt.xlabel('Window Size [s]', fontsize=10)
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # plt.subplots_adjust(wspace=0.3, hspace=0.6)
    # plt.tight_layout(rect=[0.02, 0.02, 0.98, 1])
    #
    # plt.show()
    #
    # x = 1
    #
    # ##################### Create contour plots with all 32.5 baseline window #####################
    # # log reg
    # figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-logreg_32')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.figure(figsize=(4, 14))
    # plt.subplot(6, 1, 1)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'Logistic Regression \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # RF
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-rf_32')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 2)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    # plt.title(f'Random Forest \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    #
    # # LDA
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-lda_32')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 3)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'Linear Discriminant Analysis \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # KNN
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-knn_32')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 4)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'K Nearest Neighbors \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # SVM
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-svm_32')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 5)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'Support Vector Machine \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # plt.xlabel('Window Size [s]', fontsize=10)
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # GBC
    # plt.subplots_adjust(wspace=0.3, hspace=0.6)
    # plt.tight_layout(rect=[0.02, 0.02, 0.98, 1])
    #
    # plt.show()
    #
    # x = 1
    #
    # ##################### Create contour plots with Initial Model Report results #####################
    # # log reg
    # figure_data = "C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\test_plots.xlsx"
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-logreg_old')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.figure(figsize=(4, 14))
    # plt.subplot(6, 1, 1)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'Logistic Regression \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # RF
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-rf_old')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 2)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    # plt.title(f'Random Forest \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    #
    # # LDA
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-lda_old')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 3)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'Linear Discriminant Analysis \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # KNN
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-knn_old')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 4)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'K Nearest Neighbors \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # # plt.xlabel('Window Size [s]')
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # SVM
    # df_swp = pd.read_excel(figure_data, sheet_name='SWP-svm_old')
    #
    # # convert df to numpy array
    # window_sizes_to_analyze = df_swp['window size'].to_numpy()
    # strides_to_analyze = df_swp['stride'].to_numpy()
    # baseline_windows_to_analyze = df_swp['Baseline'].to_numpy()
    # f1 = df_swp['f1'].to_numpy()
    #
    # x = window_sizes_to_analyze
    # y = strides_to_analyze
    #
    # # Find index of max f1 score
    # max_index = np.argmax(f1)
    #
    # # Determine the optimal x and y value to plot
    # optimal_x = x[max_index]
    # optimal_y = y[max_index]
    #
    # plt.subplot(6, 1, 5)
    # contour = plt.tricontourf(x, y, f1, cmap="viridis")
    # plt.colorbar(contour, label='F1 Score')
    # plt.plot(optimal_x, optimal_y, 'o', color='red', markersize=16)
    # plt.title(f'Support Vector Machine \n Baseline Window = {baseline_windows_to_analyze[0]} s', fontsize=10)
    # plt.xlabel('Window Size [s]', fontsize=10)
    # plt.ylabel('Stride [s]', fontsize=10)
    #
    # # GBC
    # plt.subplots_adjust(wspace=0.3, hspace=0.6)
    # plt.tight_layout(rect=[0.02, 0.02, 0.98, 1])
    #
    # plt.show()




    x = 1