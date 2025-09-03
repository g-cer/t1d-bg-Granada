import os
import numpy as np
import pandas as pd
import pickle
from typing import Tuple, Dict, Any

# Import existing functions
from training.split_data import load_splits, rescale_data

# Default configuration
DEFAULT_CONFIG = {
    "sample_sizes": {
        "background": 500,  # For SHAP explainer background
        "explanation": 250,  # For generating explanations
    },
    "random_seed": 42,
    "splits_dir": "training/splits",
    "cache_dir": "outputs/shared_sampling_cache",
    "glucose_thresholds": {
        "hypo": 70.0,  # mg/dL - below this is hypoglycemia
        "hyper": 180.0,  # mg/dL - above this is hyperglycemia
    },
}


class SharedSamplingManager:
    """
    Manages consistent data sampling across different model types for SHAP analysis.
    Ensures reproducible and comparable results between LSTM and XGBoost models.
    """

    def __init__(self, config=None):
        """Initialize the sampling manager with configuration"""
        self.config = config or DEFAULT_CONFIG.copy()
        self.cache_dir = self.config["cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)

        # Cache for loaded data
        self._data_cache = {}
        self._indices_cache = {}

    def _get_cache_filename(self, cache_type: str) -> str:
        """Get the cache filename for a specific type of data"""
        seed = self.config["random_seed"]
        bg_size = self.config["sample_sizes"]["background"]
        exp_size = self.config["sample_sizes"]["explanation"]
        return f"{cache_type}_seed{seed}_bg{bg_size}_exp{exp_size}.pkl"

    def _save_to_cache(self, data: Any, cache_type: str):
        """Save data to cache file"""
        cache_file = os.path.join(self.cache_dir, self._get_cache_filename(cache_type))
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {cache_type} to cache: {cache_file}")

    def _load_from_cache(self, cache_type: str) -> Any:
        """Load data from cache file if it exists"""
        cache_file = os.path.join(self.cache_dir, self._get_cache_filename(cache_type))
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {cache_type} from cache: {cache_file}")
            return data
        return None

    def load_data_splits(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
        """Load data splits with caching"""
        cache_key = "data_splits"

        if cache_key not in self._data_cache:
            # Try to load from disk cache first
            cached_data = self._load_from_cache(cache_key)

            if cached_data is not None:
                self._data_cache[cache_key] = cached_data
            else:
                # Load fresh data
                splits_dir = self.config["splits_dir"]
                train_set, val_set, test_set, X_cols, y_cols = load_splits(splits_dir)

                # Combine datasets for comprehensive analysis
                combined_data = pd.concat(
                    [train_set, val_set, test_set], ignore_index=True
                )

                data_splits = {
                    "train_set": train_set,
                    "val_set": val_set,
                    "test_set": test_set,
                    "combined_data": combined_data,
                    "X_cols": X_cols,
                    "y_cols": y_cols,
                }

                self._data_cache[cache_key] = data_splits
                self._save_to_cache(data_splits, cache_key)

        data = self._data_cache[cache_key]
        return (
            data["train_set"],
            data["val_set"],
            data["test_set"],
            data["combined_data"],
            data["X_cols"],
            data["y_cols"],
        )

    def get_sampling_indices(self) -> Dict[str, np.ndarray]:
        """Get consistent sampling indices for background and explanation data"""
        cache_key = "sampling_indices"

        if cache_key not in self._indices_cache:
            # Try to load from disk cache first
            cached_indices = self._load_from_cache(cache_key)

            if cached_indices is not None:
                self._indices_cache[cache_key] = cached_indices
            else:
                # Generate fresh indices
                train_set, val_set, test_set, combined_data, X_cols, y_cols = (
                    self.load_data_splits()
                )

                # Set random seed for reproducibility
                np.random.seed(self.config["random_seed"])

                # Sample background data indices from combined dataset
                background_indices = np.random.choice(
                    len(combined_data),
                    size=min(
                        self.config["sample_sizes"]["background"], len(combined_data)
                    ),
                    replace=False,
                )

                # Sample explanation data indices from validation set
                explanation_indices = np.random.choice(
                    len(val_set),
                    size=min(self.config["sample_sizes"]["explanation"], len(val_set)),
                    replace=False,
                )

                indices = {
                    "background_indices": background_indices,
                    "explanation_indices": explanation_indices,
                }

                self._indices_cache[cache_key] = indices
                self._save_to_cache(indices, cache_key)

                print(f"Generated sampling indices:")
                print(
                    f"  Background samples: {len(background_indices)} from {len(combined_data)} total"
                )
                print(
                    f"  Explanation samples: {len(explanation_indices)} from {len(val_set)} validation"
                )

        return self._indices_cache[cache_key]

    def get_sampled_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
        """Get the consistently sampled background and explanation data"""
        # Load data splits
        train_set, val_set, test_set, combined_data, X_cols, y_cols = (
            self.load_data_splits()
        )

        # Get sampling indices
        indices = self.get_sampling_indices()
        background_indices = indices["background_indices"]
        explanation_indices = indices["explanation_indices"]

        # Extract sampled data
        background_data = combined_data.iloc[background_indices].copy()
        explanation_data = val_set.iloc[explanation_indices].copy()

        print(f"Sampled data shapes:")
        print(f"  Background data: {background_data.shape}")
        print(f"  Explanation data: {explanation_data.shape}")
        print(f"  Features: {len(X_cols)}")

        return background_data, explanation_data, combined_data, X_cols, y_cols

    def get_predefined_clinical_cases(self) -> Dict[str, int]:
        """
        Get predefined clinical case indices that are consistent across all models.
        These are selected once based on true glucose values only, not predictions.

        Returns:
            Dictionary with clinical case indices
        """
        cache_key = "predefined_clinical_cases"

        if cache_key not in self._indices_cache:
            # Try to load from disk cache first
            cached_cases = self._load_from_cache(cache_key)

            if cached_cases is not None:
                self._indices_cache[cache_key] = cached_cases
            else:
                # Generate predefined clinical cases
                _, explanation_data, _, _, y_cols = self.get_sampled_data()

                # Get true values from explanation data
                y_true_scaled = explanation_data[y_cols[-1]].values
                y_true = rescale_data(
                    pd.DataFrame({"target": y_true_scaled}), ["target"]
                )["target"].values

                hypo_threshold = self.config["glucose_thresholds"]["hypo"]
                hyper_threshold = self.config["glucose_thresholds"]["hyper"]

                print(f"\nSelecting predefined clinical cases:")
                print(f"  Hypoglycemia: < {hypo_threshold} mg/dL")
                print(f"  Normoglycemia: {hypo_threshold} - {hyper_threshold} mg/dL")
                print(f"  Hyperglycemia: > {hyper_threshold} mg/dL")

                # Find cases based on TRUE VALUES only
                hypo_mask = y_true < hypo_threshold
                normal_mask = (y_true >= hypo_threshold) & (y_true <= hyper_threshold)
                hyper_mask = y_true > hyper_threshold

                hypo_indices = np.where(hypo_mask)[0]
                normal_indices = np.where(normal_mask)[0]
                hyper_indices = np.where(hyper_mask)[0]

                print(
                    f"  Available cases: {len(hypo_indices)} hypo, {len(normal_indices)} normal, {len(hyper_indices)} hyper"
                )

                # Select specific representative cases based on clinical interest
                predefined_cases = {}

                if len(hypo_indices) > 0:
                    # For hypo: select case closest to 60 mg/dL (severe hypoglycemia)
                    target_hypo = 60.0
                    hypo_distances = np.abs(y_true[hypo_indices] - target_hypo)
                    best_hypo_idx = np.argmin(hypo_distances)
                    predefined_cases["hypo"] = hypo_indices[best_hypo_idx]
                    print(
                        f"  Selected hypo case (index {predefined_cases['hypo']}): True={y_true[predefined_cases['hypo']]:.1f} mg/dL"
                    )

                if len(normal_indices) > 0:
                    # For normal: select case closest to 125 mg/dL (normal center)
                    target_normal = 125.0
                    normal_distances = np.abs(y_true[normal_indices] - target_normal)
                    best_normal_idx = np.argmin(normal_distances)
                    predefined_cases["normal"] = normal_indices[best_normal_idx]
                    print(
                        f"  Selected normal case (index {predefined_cases['normal']}): True={y_true[predefined_cases['normal']]:.1f} mg/dL"
                    )

                if len(hyper_indices) > 0:
                    # For hyper: select case closest to 250 mg/dL (moderate hyperglycemia)
                    target_hyper = 250.0
                    hyper_distances = np.abs(y_true[hyper_indices] - target_hyper)
                    best_hyper_idx = np.argmin(hyper_distances)
                    predefined_cases["hyper"] = hyper_indices[best_hyper_idx]
                    print(
                        f"  Selected hyper case (index {predefined_cases['hyper']}): True={y_true[predefined_cases['hyper']]:.1f} mg/dL"
                    )

                self._indices_cache[cache_key] = predefined_cases
                self._save_to_cache(predefined_cases, cache_key)

        return self._indices_cache[cache_key]

    def find_clinical_cases(
        self, predictions: np.ndarray, true_values: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get predefined clinical cases with current model predictions.
        Uses the same case indices for all models to ensure consistency.

        Args:
            predictions: Model predictions (rescaled to mg/dL)
            true_values: True glucose values (rescaled to mg/dL)

        Returns:
            Dictionary with clinical cases info including current model predictions
        """
        # Get predefined case indices
        predefined_indices = self.get_predefined_clinical_cases()

        print(f"\nUsing predefined clinical cases for consistent comparison:")

        clinical_cases = {}

        for case_type, case_idx in predefined_indices.items():
            prediction_val = predictions[case_idx]
            true_val = true_values[case_idx]
            error = abs(prediction_val - true_val)

            clinical_cases[case_type] = {
                "index": case_idx,
                "prediction": prediction_val,
                "true_value": true_val,
                "error": error,
            }

            case_name = {
                "hypo": "Hypoglycemia",
                "normal": "Normoglycemia",
                "hyper": "Hyperglycemia",
            }[case_type]
            print(
                f"  {case_name} case (index {case_idx}): Pred={prediction_val:.1f}, True={true_val:.1f} mg/dL"
            )

    def get_clinical_case_details(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about the predefined clinical cases.

        Returns:
            Dictionary with detailed case information
        """
        # Get predefined case indices
        predefined_indices = self.get_predefined_clinical_cases()

        # Get explanation data
        _, explanation_data, _, X_cols, y_cols = self.get_sampled_data()

        # Get true values
        y_true_scaled = explanation_data[y_cols[-1]].values
        y_true = rescale_data(pd.DataFrame({"target": y_true_scaled}), ["target"])[
            "target"
        ].values

        # Get feature values (rescaled)
        X_explanation_scaled = explanation_data[X_cols]
        X_explanation_rescaled = rescale_data(X_explanation_scaled, X_cols)

        case_details = {}
        case_names = {
            "hypo": "Hypoglycemia",
            "normal": "Normoglycemia",
            "hyper": "Hyperglycemia",
        }

        for case_type, case_idx in predefined_indices.items():
            case_row = explanation_data.iloc[case_idx]

            case_details[case_type] = {
                "name": case_names[case_type],
                "index": case_idx,
                "patient_id": case_row["Patient_ID"],
                "timestamp": case_row["Timestamp"],
                "true_value": y_true[case_idx],
                "features_scaled": X_explanation_scaled.iloc[case_idx].to_dict(),
                "features_rescaled": X_explanation_rescaled.iloc[case_idx].to_dict(),
            }

        return case_details

    def print_clinical_cases_info(self):
        """Print detailed information about the predefined clinical cases"""
        print("\n" + "=" * 60)
        print("PREDEFINED CLINICAL CASES INFORMATION")
        print("=" * 60)

        case_details = self.get_clinical_case_details()

        for case_type, details in case_details.items():
            print(f"\n{details['name']} Case:")
            print(f"  Index: {details['index']}")
            print(f"  Patient ID: {details['patient_id']}")
            print(f"  Timestamp: {details['timestamp']}")
            print(f"  True glucose value: {details['true_value']:.1f} mg/dL")
            print(f"  Feature values (rescaled to mg/dL):")
            for feature, value in details["features_rescaled"].items():
                print(f"    {feature}: {value:.1f} mg/dL")

        print("=" * 60)
        """Print information about the current sampling configuration"""
        print("=" * 60)
        print("SHARED SAMPLING CONFIGURATION")
        print("=" * 60)
        print(f"Random seed: {self.config['random_seed']}")
        print(f"Background samples: {self.config['sample_sizes']['background']}")
        print(f"Explanation samples: {self.config['sample_sizes']['explanation']}")
        print(
            f"Hypoglycemia threshold: {self.config['glucose_thresholds']['hypo']} mg/dL"
        )
        print(
            f"Hyperglycemia threshold: {self.config['glucose_thresholds']['hyper']} mg/dL"
        )
        print(f"Cache directory: {self.cache_dir}")
        print("=" * 60)

    def print_sampling_info(self):
        """Print information about the current sampling configuration"""
        print("=" * 60)
        print("SHARED SAMPLING CONFIGURATION")
        print("=" * 60)
        print(f"Random seed: {self.config['random_seed']}")
        print(f"Background samples: {self.config['sample_sizes']['background']}")
        print(f"Explanation samples: {self.config['sample_sizes']['explanation']}")
        print(
            f"Hypoglycemia threshold: {self.config['glucose_thresholds']['hypo']} mg/dL"
        )
        print(
            f"Hyperglycemia threshold: {self.config['glucose_thresholds']['hyper']} mg/dL"
        )
        print(f"Cache directory: {self.cache_dir}")
        print("=" * 60)


def create_shared_sampling_manager(config_override=None) -> SharedSamplingManager:
    """
    Factory function to create a shared sampling manager.

    Args:
        config_override: Dictionary to override default configuration

    Returns:
        SharedSamplingManager instance
    """
    config = DEFAULT_CONFIG.copy()
    if config_override:
        # Deep merge the configuration
        for key, value in config_override.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return SharedSamplingManager(config)


if __name__ == "__main__":
    # Test the shared sampling manager
    print("Testing Shared Sampling Manager...")

    manager = create_shared_sampling_manager()
    manager.print_sampling_info()

    # Test data loading
    background_data, explanation_data, combined_data, X_cols, y_cols = (
        manager.get_sampled_data()
    )

    print(f"\nTest completed successfully!")
    print(f"Background data: {background_data.shape}")
    print(f"Explanation data: {explanation_data.shape}")
    print(f"Features: {X_cols}")

    # Test predefined clinical cases
    print("\n" + "=" * 60)
    print("TESTING PREDEFINED CLINICAL CASES")
    print("=" * 60)

    # Get predefined cases
    predefined_cases = manager.get_predefined_clinical_cases()
    print(f"Predefined case indices: {predefined_cases}")

    # Print detailed case information
    manager.print_clinical_cases_info()

    # Test with mock predictions
    print("\nTesting with mock predictions...")
    y_true_scaled = explanation_data[y_cols[-1]].values
    y_true = rescale_data(pd.DataFrame({"target": y_true_scaled}), ["target"])[
        "target"
    ].values

    # Create mock predictions
    np.random.seed(42)
    mock_predictions = np.random.normal(150, 50, len(y_true))
    mock_predictions = np.clip(mock_predictions, 40, 400)

    # Test clinical cases with mock predictions
    clinical_cases = manager.get_predefined_clinical_cases()

    print("\nPredefined clinical cases test:")
    for case_type, case_index in clinical_cases.items():
        pred = mock_predictions[case_index]
        true_val = y_true[case_index]
        error = abs(pred - true_val)
        print(
            f"  {case_type}: Index={case_index}, Pred={pred:.1f}, True={true_val:.1f}, Error={error:.1f}"
        )

    print(f"\n✅ Predefined clinical cases test completed!")
