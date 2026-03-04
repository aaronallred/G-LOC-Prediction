class TraditionalDataManager:
    """Data manager for traditional data pipeline."""
    def __init__(self, data_path: str = "../data/", testing: bool = False, random_seed: int = 42, use_reduced_dataset: bool = False) -> None:
        self.data_path = data_path
        self._data_locations = None
        self.testing = testing
        self.random_seed = random_seed
        self.use_reduced_dataset = use_reduced_dataset

    def get_data(self, backstep, data_rate, classifier_type, model_type, select_features):
        """Return data for a given set of parameters."""
        # This method would typically load and return the appropriate dataset
        # based on the provided parameters.
        return {
            "backstep": backstep,
            "data_rate": data_rate,
            "classifier_type": classifier_type,
            "model_type": model_type,
            "select_features": select_features
        }
    
    def _get_hyperparameters_by_classifier(self, classifier_type):
        """Return hyperparameters for a given classifier type."""
        if classifier_type == 'logreg':
            # Specifying Methods from Sequential optimization
            baseline_window = 5  # seconds - PULLED FROM NIKKI PAPER
            window_size = 12.5 # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25 # seconds - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'lasso' #- PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8'] #- PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 5  # -For imputation PULLED FROM NIKKI PAPER

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 8 # ~ 0.1 hit to f1 score



        if classifier_type == 'RF':
            # Specifying Methods from Sequential optimization
            baseline_window = 18.75  # seconds - PULLED FROM NIKKI PAPER
            window_size = 7.5  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'none'  # - PULLED FROM NIKKI PAPER
            threshold = 30  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 5 # ~ 0.1 hit to f1 score


        if classifier_type == 'LDA':
            # Specifying Methods from Sequential optimization
            baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
            window_size = 15  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 10 # ~ 0.3 hit to f1 score


        if classifier_type == 'SVM':
            # Specifying Methods from Sequential optimization
            baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
            window_size = 15  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'ridge'  # - PULLED FROM NIKKI PAPER
            threshold = 10  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # - For imputation PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 8 # ~ 0.2 hit to f1 score


        if classifier_type == 'EGB':
            # Specifying Methods from Sequential optimization
            baseline_window = 46.25  # seconds - PULLED FROM NIKKI PAPER
            window_size = 12.5  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'lasso'  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2','v5','v6','v7','v8']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'none'  # - PULLED FROM NIKKI PAPER
            impute_type = 1  # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 3  # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            #window_size = 8 # ~ 0.1 hit to f1 score


        if classifier_type == 'KNN':
            # Specifying Methods from Sequential optimization
            baseline_window = 32.5  # seconds - PULLED FROM NIKKI PAPER
            window_size = 15  # seconds - PULLED FROM NIKKI PAPER
            stride = 0.25  # seconds - PULLED FROM NIKKI PAPER
            feature_reduction_type = 'performance'  # - PULLED FROM NIKKI PAPER
            baseline_methods_to_use = ['v0', 'v1', 'v2']  # - PULLED FROM NIKKI PAPER
            imbalance_type = 'ros' # - PULLED FROM NIKKI PAPER
            impute_type = 1 # - PULLED FROM NIKKI PAPER, 1 signifies yes KNN imputation used
            n_neighbors = 5 # -For imputation PULLED FROM NIKKI PAPER
            # Code for loading txt

            ## Investigating different windows per EVAN ANDERSON
            # window_size = 12 # ~ 0.1 hit to f1 score

        return baseline_window, window_size, stride, feature_reduction_type, baseline_methods_to_use, imbalance_type, impute_type, n_neighbors