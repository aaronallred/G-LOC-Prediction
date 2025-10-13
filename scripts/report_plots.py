import pickle
import pandas as pd
from GLOC_visualization import *
import numpy as np

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


############################# IMPORT & COMBINE ALL FOLDS FOR KNN ##########################
## EXPLICIT ##
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_0.pkl', 'rb') as file:
    cv_knn_explicit = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_1.pkl', 'rb') as file:
    cv_knn_explicit_1 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_2.pkl', 'rb') as file:
    cv_knn_explicit_2 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_3.pkl', 'rb') as file:
    cv_knn_explicit_3 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_4.pkl', 'rb') as file:
    cv_knn_explicit_4 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_5.pkl', 'rb') as file:
    cv_knn_explicit_5 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_6.pkl', 'rb') as file:
    cv_knn_explicit_6 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_7.pkl', 'rb') as file:
    cv_knn_explicit_7 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_8.pkl', 'rb') as file:
    cv_knn_explicit_8 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_explicit_9.pkl', 'rb') as file:
    cv_knn_explicit_9 = pickle.load(file)


cv_knn_explicit.update(cv_knn_explicit_1)
cv_knn_explicit.update(cv_knn_explicit_2)
cv_knn_explicit.update(cv_knn_explicit_3)
cv_knn_explicit.update(cv_knn_explicit_4)
cv_knn_explicit.update(cv_knn_explicit_5)
cv_knn_explicit.update(cv_knn_explicit_6)
cv_knn_explicit.update(cv_knn_explicit_7)
cv_knn_explicit.update(cv_knn_explicit_8)
cv_knn_explicit.update(cv_knn_explicit_9)

## IMPLICIT ##
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_0.pkl', 'rb') as file:
    cv_knn_implicit = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_1.pkl', 'rb') as file:
    cv_knn_implicit_1 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_2.pkl', 'rb') as file:
    cv_knn_implicit_2 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_3.pkl', 'rb') as file:
    cv_knn_implicit_3 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_4.pkl', 'rb') as file:
    cv_knn_implicit_4 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_5.pkl', 'rb') as file:
    cv_knn_implicit_5 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_6.pkl', 'rb') as file:
    cv_knn_implicit_6 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_7.pkl', 'rb') as file:
    cv_knn_implicit_7 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_8.pkl', 'rb') as file:
    cv_knn_implicit_8 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_knn_implicit_9.pkl', 'rb') as file:
    cv_knn_implicit_9 = pickle.load(file)

cv_knn_implicit.update(cv_knn_implicit_1)
cv_knn_implicit.update(cv_knn_implicit_2)
cv_knn_implicit.update(cv_knn_implicit_3)
cv_knn_implicit.update(cv_knn_implicit_4)
cv_knn_implicit.update(cv_knn_implicit_5)
cv_knn_implicit.update(cv_knn_implicit_6)
cv_knn_implicit.update(cv_knn_implicit_7)
cv_knn_implicit.update(cv_knn_implicit_8)
cv_knn_implicit.update(cv_knn_implicit_9)


############################# IMPORT & COMBINE ALL FOLDS FOR LOGREG ##########################
## EXPLICIT ##
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_0.pkl', 'rb') as file:
    cv_logreg_explicit = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_1.pkl', 'rb') as file:
    cv_logreg_explicit_1 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_2.pkl', 'rb') as file:
    cv_logreg_explicit_2 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_3.pkl', 'rb') as file:
    cv_logreg_explicit_3 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_4.pkl', 'rb') as file:
    cv_logreg_explicit_4 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_5.pkl', 'rb') as file:
    cv_logreg_explicit_5 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_6.pkl', 'rb') as file:
    cv_logreg_explicit_6 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_7.pkl', 'rb') as file:
    cv_logreg_explicit_7 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_8.pkl', 'rb') as file:
    cv_logreg_explicit_8 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_explicit_9.pkl', 'rb') as file:
    cv_logreg_explicit_9 = pickle.load(file)

cv_logreg_explicit.update(cv_logreg_explicit_1)
cv_logreg_explicit.update(cv_logreg_explicit_2)
cv_logreg_explicit.update(cv_logreg_explicit_3)
cv_logreg_explicit.update(cv_logreg_explicit_4)
cv_logreg_explicit.update(cv_logreg_explicit_5)
cv_logreg_explicit.update(cv_logreg_explicit_6)
cv_logreg_explicit.update(cv_logreg_explicit_7)
cv_logreg_explicit.update(cv_logreg_explicit_8)
cv_logreg_explicit.update(cv_logreg_explicit_9)

## IMPLICIT ##
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_0.pkl', 'rb') as file:
    cv_logreg_implicit = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_1.pkl', 'rb') as file:
    cv_logreg_implicit_1 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_2.pkl', 'rb') as file:
    cv_logreg_implicit_2 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_3.pkl', 'rb') as file:
    cv_logreg_implicit_3 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_4.pkl', 'rb') as file:
    cv_logreg_implicit_4 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_5.pkl', 'rb') as file:
    cv_logreg_implicit_5 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_6.pkl', 'rb') as file:
    cv_logreg_implicit_6 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_7.pkl', 'rb') as file:
    cv_logreg_implicit_7 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_8.pkl', 'rb') as file:
    cv_logreg_implicit_8 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_logreg_implicit_9.pkl', 'rb') as file:
    cv_logreg_implicit_9 = pickle.load(file)

cv_logreg_implicit.update(cv_logreg_implicit_1)
cv_logreg_implicit.update(cv_logreg_implicit_2)
cv_logreg_implicit.update(cv_logreg_implicit_3)
cv_logreg_implicit.update(cv_logreg_implicit_4)
cv_logreg_implicit.update(cv_logreg_implicit_5)
cv_logreg_implicit.update(cv_logreg_implicit_6)
cv_logreg_implicit.update(cv_logreg_implicit_7)
cv_logreg_implicit.update(cv_logreg_implicit_8)
cv_logreg_implicit.update(cv_logreg_implicit_9)


############################# IMPORT & COMBINE ALL FOLDS FOR GRAD BOOST ##########################
## EXPLICIT ##
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_0.pkl', 'rb') as file:
    cv_grad_boost_explicit = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_1.pkl', 'rb') as file:
    cv_grad_boost_explicit_1 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_2.pkl', 'rb') as file:
    cv_grad_boost_explicit_2 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_3.pkl', 'rb') as file:
    cv_grad_boost_explicit_3 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_4.pkl', 'rb') as file:
    cv_grad_boost_explicit_4 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_5.pkl', 'rb') as file:
    cv_grad_boost_explicit_5 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_6.pkl', 'rb') as file:
    cv_grad_boost_explicit_6 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_7.pkl', 'rb') as file:
    cv_grad_boost_explicit_7 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_8.pkl', 'rb') as file:
    cv_grad_boost_explicit_8 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_explicit_9.pkl', 'rb') as file:
    cv_grad_boost_explicit_9 = pickle.load(file)

cv_grad_boost_explicit.update(cv_grad_boost_explicit_1)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_2)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_3)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_4)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_5)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_6)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_7)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_8)
cv_grad_boost_explicit.update(cv_grad_boost_explicit_9)

## IMPLICIT ##
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_0.pkl', 'rb') as file:
    cv_grad_boost_implicit = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_1.pkl', 'rb') as file:
    cv_grad_boost_implicit_1 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_2.pkl', 'rb') as file:
    cv_grad_boost_implicit_2 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_3.pkl', 'rb') as file:
    cv_grad_boost_implicit_3 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_4.pkl', 'rb') as file:
    cv_grad_boost_implicit_4 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_5.pkl', 'rb') as file:
    cv_grad_boost_implicit_5 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_6.pkl', 'rb') as file:
    cv_grad_boost_implicit_6 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_7.pkl', 'rb') as file:
    cv_grad_boost_implicit_7 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_8.pkl', 'rb') as file:
    cv_grad_boost_implicit_8 = pickle.load(file)

with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_grad_boost_implicit_9.pkl', 'rb') as file:
    cv_grad_boost_implicit_9 = pickle.load(file)

cv_grad_boost_implicit.update(cv_grad_boost_implicit_1)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_2)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_3)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_4)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_5)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_6)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_7)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_8)
cv_grad_boost_implicit.update(cv_grad_boost_implicit_9)

############################# LOGREG IMPLICIT ##########################
# with open('C:\\Users\\nicol\\Downloads\\CrossValidation_logreg_implicit.pkl', 'rb') as file:
#     cv_logreg_implicit = pickle.load(file)

############################# RF IMPLICIT ##########################
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_rf_implicit.pkl', 'rb') as file:
    cv_rf_implicit = pickle.load(file)

############################# LDA IMPLICIT ##########################
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_lda_implicit.pkl', 'rb') as file:
    cv_lda_implicit = pickle.load(file)

############################# KNN IMPLICIT ##########################
# with open('C:\\Users\\nicol\\Downloads\\CrossValidation_knn_implicit.pkl', 'rb') as file:
#     cv_knn_implicit = pickle.load(file)

############################# SVM IMPLICIT ##########################
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_svm_implicit.pkl', 'rb') as file:
    cv_svm_implicit = pickle.load(file)

############################# GRAD BOOST IMPLICIT ##########################
# with open('C:\\Users\\nicol\\Downloads\\CrossValidation_grad_boost_implicit.pkl', 'rb') as file:
#     cv_grad_boost_implicit = pickle.load(file)

############################# LOGREG EXPLICIT ##########################
# with open('C:\\Users\\nicol\\Downloads\\CrossValidation_logreg_explicit.pkl', 'rb') as file:
#     cv_logreg_explicit = pickle.load(file)

############################# RF EXPLICIT ##########################
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_rf_explicit.pkl', 'rb') as file:
    cv_rf_explicit = pickle.load(file)

############################# LDA EXPLICIT ##########################
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_lda_explicit.pkl', 'rb') as file:
    cv_lda_explicit = pickle.load(file)

############################# KNN EXPLICIT ##########################
# with open('C:\\Users\\nicol\\Downloads\\CrossValidation_knn_explicit.pkl', 'rb') as file:
#     cv_knn_explicit = pickle.load(file)

############################# SVM EXPLICIT ##########################
with open('C:\\Users\\nicol\\OneDrive\\Documents\\Research\\gloc\\G-LOC-Prediction\\CrossValidation\\CrossValidation_svm_explicit.pkl', 'rb') as file:
    cv_svm_explicit = pickle.load(file)

############################# GRAD BOOST EXPLICIT ##########################
# with open('C:\\Users\\nicol\\Downloads\\CrossValidation_grad_boost_explicit.pkl', 'rb') as file:
#     cv_grad_boost_explicit = pickle.load(file)


######################################## MAKE FINAL EXPLICIT DICTIONARY ####################################
all_cv_explicit = {}
all_keys = set(cv_rf_explicit)
for key in all_keys:
    if (key in cv_logreg_explicit and key in cv_rf_explicit and key in cv_lda_explicit and key in cv_knn_explicit
            and key in cv_svm_explicit and key in cv_grad_boost_explicit):
        # Concatenate DataFrames for common keys
        all_cv_explicit[key] = pd.concat([cv_logreg_explicit[key], cv_rf_explicit[key], cv_lda_explicit[key],
                                          cv_knn_explicit[key], cv_svm_explicit[key], cv_grad_boost_explicit[key]])

    # Missing EGB
    elif (key in cv_logreg_explicit and key in cv_rf_explicit and key in cv_lda_explicit and key in cv_knn_explicit
            and key in cv_svm_explicit):
        all_cv_explicit[key] = pd.concat([cv_logreg_explicit[key], cv_rf_explicit[key], cv_lda_explicit[key],
                                          cv_knn_explicit[key], cv_svm_explicit[key]])

    # MISSING KNN, LOGREG, EGB
    elif key in cv_rf_explicit and key in cv_lda_explicit and key in cv_svm_explicit:
        # Concatenate DataFrames for common keys
        all_cv_explicit[key] = pd.concat([cv_rf_explicit[key], cv_lda_explicit[key], cv_svm_explicit[key]])

# plot_cross_val(all_cv_explicit)
# plot_cross_val_sp(all_cv_explicit)

######################################## MAKE FINAL IMPLICIT DICTIONARY ####################################
all_cv_implicit = {}
all_keys = set(cv_rf_implicit)
for key in all_keys:
    if (key in cv_logreg_implicit and key in cv_rf_implicit and key in cv_lda_implicit and key in cv_knn_implicit
            and key in cv_svm_implicit and key in cv_grad_boost_implicit):
        # Concatenate DataFrames for common keys
        all_cv_implicit[key] = pd.concat([cv_logreg_implicit[key], cv_rf_implicit[key], cv_lda_implicit[key],
                                          cv_knn_implicit[key], cv_svm_implicit[key], cv_grad_boost_implicit[key]])

    # Missing EGB
    elif (key in cv_logreg_implicit and key in cv_rf_implicit and key in cv_lda_implicit and key in cv_knn_implicit
            and key in cv_svm_implicit):
        all_cv_implicit[key] = pd.concat([cv_logreg_implicit[key], cv_rf_implicit[key], cv_lda_implicit[key],
                                          cv_knn_implicit[key], cv_svm_implicit[key]])

    # Missing LOGREG
    elif (key in cv_rf_implicit and key in cv_lda_implicit and key in cv_knn_implicit
        and key in cv_svm_implicit and key in cv_grad_boost_implicit):
        # Concatenate DataFrames for common keys
        all_cv_implicit[key] = pd.concat([cv_rf_implicit[key], cv_lda_implicit[key],
                                          cv_knn_implicit[key], cv_svm_implicit[key], cv_grad_boost_implicit[key]])

    # MISSING KNN, LOGREG
    elif key in cv_rf_implicit and key in cv_lda_implicit and key in cv_svm_implicit and key in cv_grad_boost_implicit:
        # Concatenate DataFrames for common keys
        all_cv_implicit[key] = pd.concat([cv_rf_implicit[key], cv_lda_implicit[key], cv_svm_implicit[key], cv_grad_boost_implicit[key]])

    # MISSING KNN, LOGREG, EGB
    elif key in cv_rf_implicit and key in cv_lda_implicit and key in cv_svm_implicit:
        # Concatenate DataFrames for common keys
        all_cv_implicit[key] = pd.concat([cv_rf_implicit[key], cv_lda_implicit[key], cv_svm_implicit[key]])

# plot_cross_val(all_cv_implicit)
# plot_cross_val_sp(all_cv_implicit)
#
# plot_cross_val(cv_knn_explicit)
# plot_cross_val(cv_knn_implicit)


###################### CREATE UNPACKED DICTIONARY PER CLASSIFIER AND MODEL TYPE #########################
combined_logreg_explicit = pd.concat(cv_logreg_explicit.values(), ignore_index=True)
combined_logreg_explicit_array = combined_logreg_explicit.to_numpy()

combined_logreg_implicit = pd.concat(cv_logreg_implicit .values(), ignore_index=True)
combined_logreg_implicit_array = combined_logreg_implicit .to_numpy()

combined_rf_explicit = pd.concat(cv_rf_explicit.values(), ignore_index=True)
combined_rf_explicit_array = combined_rf_explicit.to_numpy()

combined_rf_implicit = pd.concat(cv_rf_implicit .values(), ignore_index=True)
combined_rf_implicit_array = combined_rf_implicit .to_numpy()

combined_lda_explicit = pd.concat(cv_lda_explicit.values(), ignore_index=True)
combined_lda_explicit_array = combined_lda_explicit.to_numpy()

combined_lda_implicit = pd.concat(cv_lda_implicit .values(), ignore_index=True)
combined_lda_implicit_array = combined_lda_implicit .to_numpy()

combined_knn_explicit = pd.concat(cv_knn_explicit.values(), ignore_index=True)
combined_knn_explicit_array = combined_knn_explicit.to_numpy()

combined_knn_implicit = pd.concat(cv_knn_implicit .values(), ignore_index=True)
combined_knn_implicit_array = combined_knn_implicit .to_numpy()

combined_svm_explicit = pd.concat(cv_svm_explicit.values(), ignore_index=True)
combined_svm_explicit_array = combined_svm_explicit.to_numpy()

combined_svm_implicit = pd.concat(cv_svm_implicit .values(), ignore_index=True)
combined_svm_implicit_array = combined_svm_implicit .to_numpy()

combined_grad_boost_explicit = pd.concat(cv_grad_boost_explicit.values(), ignore_index=True)
combined_grad_boost_explicit_array = combined_grad_boost_explicit.to_numpy()

combined_grad_boost_implicit = pd.concat(cv_grad_boost_implicit .values(), ignore_index=True)
combined_grad_boost_implicit_array = combined_grad_boost_implicit .to_numpy()


# COMPUTE MEAN AND STANDARD DEVIATION #
mean_logreg_explicit = np.mean(combined_logreg_explicit_array, axis=0)
std_dev_logreg_explicit = np.std(combined_logreg_explicit_array, axis=0)
median_dev_logreg_explicit = np.median(combined_logreg_explicit_array, axis=0)

mean_logreg_implicit = np.mean(combined_logreg_implicit_array, axis=0)
std_dev_logreg_implicit = np.std(combined_logreg_implicit_array, axis=0)
median_dev_logreg_implicit = np.median(combined_logreg_implicit_array, axis=0)

mean_rf_explicit = np.mean(combined_rf_explicit_array, axis=0)
std_dev_rf_explicit = np.std(combined_rf_explicit_array, axis=0)
median_dev_rf_explicit = np.median(combined_rf_explicit_array, axis=0)

mean_rf_implicit = np.mean(combined_rf_implicit_array, axis=0)
std_dev_rf_implicit = np.std(combined_rf_implicit_array, axis=0)
median_dev_rf_implicit = np.median(combined_rf_implicit_array, axis=0)

mean_lda_explicit = np.mean(combined_lda_explicit_array, axis=0)
std_dev_lda_explicit = np.std(combined_lda_explicit_array, axis=0)
median_dev_lda_explicit = np.median(combined_lda_explicit_array, axis=0)

mean_lda_implicit = np.mean(combined_lda_implicit_array, axis=0)
std_dev_lda_implicit = np.std(combined_lda_implicit_array, axis=0)
median_dev_lda_implicit = np.median(combined_lda_implicit_array, axis=0)

mean_knn_explicit = np.mean(combined_knn_explicit_array, axis=0)
std_dev_knn_explicit = np.std(combined_knn_explicit_array, axis=0)
median_dev_knn_explicit = np.median(combined_knn_explicit_array, axis=0)

mean_knn_implicit = np.mean(combined_knn_implicit_array, axis=0)
std_dev_knn_implicit = np.std(combined_knn_implicit_array, axis=0)
median_dev_knn_implicit = np.median(combined_knn_implicit_array, axis=0)

mean_svm_explicit = np.mean(combined_svm_explicit_array, axis=0)
std_dev_svm_explicit = np.std(combined_svm_explicit_array, axis=0)
median_dev_svm_explicit = np.median(combined_svm_explicit_array, axis=0)

mean_svm_implicit = np.mean(combined_svm_implicit_array, axis=0)
std_dev_svm_implicit = np.std(combined_svm_implicit_array, axis=0)
median_dev_svm_implicit = np.median(combined_svm_implicit_array, axis=0)

mean_grad_boost_explicit = np.mean(combined_grad_boost_explicit_array, axis=0)
std_dev_grad_boost_explicit = np.std(combined_grad_boost_explicit_array, axis=0)
median_dev_grad_boost_explicit = np.median(combined_grad_boost_explicit_array, axis=0)

mean_grad_boost_implicit = np.mean(combined_grad_boost_implicit_array, axis=0)
std_dev_grad_boost_implicit = np.std(combined_grad_boost_implicit_array, axis=0)
median_dev_grad_boost_implicit = np.median(combined_grad_boost_implicit_array, axis=0)


br_ex = bridget_recall_specificity_balanced_accuracy(all_cv_explicit)
br_im = bridget_recall_specificity_balanced_accuracy(all_cv_implicit)

x = 1