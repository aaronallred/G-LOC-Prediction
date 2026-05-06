from pathlib import Path
import pickle
import shap
import matplotlib.pyplot as plt


def plot_all_shap(
    subfolder_name,
    fold_num,
    classifier="KNN",
    plot_types=("violin", "beeswarm", "bar"),
    base_dir=None,
    explanation=None,
):
    """
    Load a SHAP Explanation object from a pickle file and generate plots.

    Parameters
    ----------
    subfolder_name : str
        Name of the subfolder (e.g., 'Explicit Complete')
    fold_num : int
        Fold number to load
    classifier : str
        Model name (default: 'KNN')
    plot_types : tuple
        Which plots to generate ('violin', 'beeswarm', 'bar')
    base_dir : Path or None
        Base directory; defaults to two levels up from this file
    """

    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]

    results_folder = base_dir / "feature_study" / subfolder_name / "streamwise"

    data_file_path = results_folder / f"shap_{classifier}_f{fold_num}.pkl"

    # # Load explanation
    # with open(data_file_path, "rb") as f:
    #     explanation = pickle.load(f)

    # Slice class 1
    if len(explanation.values.shape) == 3:
        class1_exp = explanation[:, :, 1]
    else:
        class1_exp = explanation

    # Reattach feature names (important after pickle load)
    class1_exp.feature_names = explanation.feature_names

    # Plot mapping
    plot_funcs = {
        "violin": shap.plots.violin,
        "beeswarm": shap.plots.beeswarm,
        "bar": shap.plots.bar,
    }

    for plot_type in plot_types:
        if plot_type not in plot_funcs:
            print(f"Skipping unknown plot type: {plot_type}")
            continue

        try:
            plt.figure()
            plot_funcs[plot_type](class1_exp, show=False)

            save_path = results_folder / f"shap_{classifier}_{plot_type}_plot_f{fold_num}.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

            print(f"Saved {plot_type} plot to: {save_path}")

        except Exception as e:
            print(f"Skipping {plot_type} plot due to error: {e}")
            plt.close()