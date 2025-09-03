import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap
import seaborn as sns

print(f"PyTorch version: {torch.__version__}")
print(f"SHAP version: {shap.__version__}")

# Import existing functions
from training.split_data import rescale_data, U_BOUND, L_BOUND
from training.pt_utils_dnn import RNNModel
from xai.shared_sampling import create_shared_sampling_manager

# Set style for plots
plt.style.use("default")
sns.set_palette("husl")

# Configuration
config = {
    "model_path": "pt_outputs/lstm.pt",
    "output_dir": "outputs",
    "plots_dir": "plots",
    "splits_dir": "training/splits",
    "shap_plots_dir": "plots/lstm/shap_analysis_kernel",
    "sample_sizes": {
        "background": 500,  # For SHAP explainer background
        "explanation": 250,  # For generating explanations
    },
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_params": {
        "variant": "lstm",
        "input_size": 1,
        "hidden_size": 75,
        "num_layers": 2,
        "dropout": 0.2,
    },
}


def print_config():
    """Print configuration details"""
    print("=" * 60)
    print("LSTM SHAP KERNEL ANALYSIS CONFIGURATION")
    print("=" * 60)
    print(f"Model path: {config['model_path']}")
    print(f"Device: {config['device']}")
    print(f"Background samples: {config['sample_sizes']['background']}")
    print(f"Explanation samples: {config['sample_sizes']['explanation']}")
    print(f"Random seed: {config['random_seed']}")
    print(f"Explainer type: KernelExplainer (model-agnostic)")
    print("=" * 60)


def setup_directories():
    """Create output directories"""
    os.makedirs(config["plots_dir"], exist_ok=True)
    os.makedirs(config["shap_plots_dir"], exist_ok=True)
    print(f"Created directories: {config['shap_plots_dir']}")


def load_lstm_model():
    """Load the trained LSTM model"""
    print("\n" + "=" * 60)
    print("LOADING LSTM MODEL")
    print("=" * 60)

    if not os.path.exists(config["model_path"]):
        raise FileNotFoundError(f"LSTM model not found at {config['model_path']}")

    # Create model instance
    model = RNNModel(
        variant=config["model_params"]["variant"],
        input_size=config["model_params"]["input_size"],
        hidden_size=config["model_params"]["hidden_size"],
        num_layers=config["model_params"]["num_layers"],
        dropout=config["model_params"]["dropout"],
    )

    # Load state dict
    model.load_state_dict(
        torch.load(config["model_path"], map_location=config["device"])
    )
    model.to(config["device"])
    model.eval()

    print(f"LSTM model loaded from {config['model_path']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    return model


def prepare_shap_data():
    """Prepare data for SHAP analysis using shared sampling"""
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR SHAP ANALYSIS (SHARED SAMPLING)")
    print("=" * 60)

    # Create shared sampling manager
    sampling_config = {
        "sample_sizes": config["sample_sizes"],
        "random_seed": config["random_seed"],
        "splits_dir": config["splits_dir"],
    }
    manager = create_shared_sampling_manager(sampling_config)
    manager.print_sampling_info()

    # Get consistently sampled data
    background_data, explanation_data, combined_data, X_cols, y_cols = (
        manager.get_sampled_data()
    )

    print(f"Data loaded successfully:")
    print(f"  Combined dataset: {combined_data.shape}")
    print(f"  Background data: {background_data.shape}")
    print(f"  Explanation data: {explanation_data.shape}")
    print(f"  Features: {len(X_cols)}")
    print(f"  Feature names: {X_cols}")

    return combined_data, background_data, explanation_data, X_cols, y_cols, manager


def create_lstm_wrapper(model, X_cols, device):
    """Create a wrapper function for SHAP that handles LSTM input format"""

    def lstm_predict(X):
        """
        Wrapper function for LSTM predictions compatible with SHAP KernelExplainer
        X: numpy array of shape (n_samples, n_features)
        Returns: numpy array of predictions
        """
        if isinstance(X, np.ndarray):
            # Convert to tensor and reshape for LSTM
            X_tensor = torch.tensor(X.astype(np.float32))
            # Reshape to (batch_size, sequence_length, input_size)
            X_tensor = X_tensor.reshape(X_tensor.shape[0], X_tensor.shape[1], 1)
        else:
            X_tensor = X

        X_tensor = X_tensor.to(device)

        with torch.no_grad():
            model.eval()
            predictions = model(X_tensor)
            # Return predictions as numpy array, flattened
            return predictions.cpu().numpy().flatten()

    return lstm_predict


def find_clinical_cases(manager):
    """
    Get predefined clinical cases using shared sampling manager for consistency
    """
    return manager.get_predefined_clinical_cases()


def generate_clinical_waterfall_plots(
    clinical_cases,
    shap_values_rescaled,
    X_explanation_rescaled,
    predictions_rescaled,
    y_true_rescaled,
    expected_value,
    X_cols,
):
    """Generate waterfall plots for clinical cases (hypo, normal, hyper)"""

    case_names = {
        "hypo": "Hypoglycemia",
        "normal": "Normoglycemia",
        "hyper": "Hyperglycemia",
    }

    case_colors = {
        "hypo": "#FF6B6B",  # Light red
        "normal": "#4ECDC4",  # Teal
        "hyper": "#FFD93D",  # Yellow
    }

    for case_type, case_index in clinical_cases.items():
        case_name = case_names[case_type]

        print(f"Generating {case_name} waterfall plot...")

        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values_rescaled[case_index],
            base_values=expected_value,
            data=X_explanation_rescaled.iloc[case_index].values,
            feature_names=X_cols,
        )

        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(explanation, show=False)

        # Customize the plot
        prediction_val = predictions_rescaled[case_index]
        true_val = y_true_rescaled[case_index]
        error = abs(prediction_val - true_val)

        # Set title with clinical context
        title = f"LSTM SHAP Explanation (KernelExplainer) - {case_name} Case\n"
        title += f"Prediction: {prediction_val:.1f} mg/dL | True Value: {true_val:.1f} mg/dL | Error: {error:.1f} mg/dL"

        plt.title(title, fontsize=14, fontweight="bold")

        # Save plot
        filename = f"lstm_shap_kernel_waterfall_{case_type}.png"
        plt.tight_layout()
        plt.savefig(
            f"{config['shap_plots_dir']}/{filename}",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Print interpretation
        print(f"\n{case_name} Case Interpretation:")
        print(f"  Baseline (expected): {expected_value:.1f} mg/dL")
        print(f"  Final prediction: {prediction_val:.1f} mg/dL")
        print(f"  Actual value: {true_val:.1f} mg/dL")
        print(f"  Prediction error: {error:.1f} mg/dL")

        # Show top contributing features
        feature_contributions = shap_values_rescaled[case_index]
        abs_contributions = np.abs(feature_contributions)
        top_indices = np.argsort(abs_contributions)[::-1][:3]

        print(f"  Top 3 contributing features:")
        for i, feat_idx in enumerate(top_indices):
            feat_name = X_cols[feat_idx]
            contribution = feature_contributions[feat_idx]
            feat_value = X_explanation_rescaled.iloc[case_index, feat_idx]
            direction = "increases" if contribution > 0 else "decreases"
            print(
                f"    {i+1}. {feat_name}: {contribution:+.1f} mg/dL ({direction} prediction)"
            )
            print(f"       Feature value: {feat_value:.1f} mg/dL")


def analyze_lstm_shap(model, background_data, explanation_data, X_cols):
    """Perform SHAP analysis for LSTM model using KernelExplainer"""
    print("\n" + "=" * 60)
    print("LSTM SHAP ANALYSIS WITH KERNELEXPLAINER")
    print("=" * 60)

    # Prepare data for LSTM wrapper (2D format for KernelExplainer)
    X_background = background_data[X_cols].values.astype(np.float32)
    X_explanation = explanation_data[X_cols].values.astype(np.float32)

    print(f"Background data shape: {X_background.shape}")
    print(f"Explanation data shape: {X_explanation.shape}")

    # Create LSTM wrapper function for model-agnostic explainer
    lstm_predict_fn = create_lstm_wrapper(model, X_cols, config["device"])

    # Test the wrapper function
    print("Testing LSTM wrapper function...")
    test_pred = lstm_predict_fn(X_background[:5])
    print(f"Test prediction shape: {test_pred.shape}")
    print(f"Test predictions: {test_pred[:3]}")

    # Create KernelExplainer (model-agnostic, works with any model)
    print("Creating KernelExplainer for LSTM...")
    print(
        "This may take a while as KernelExplainer is slower than gradient-based methods..."
    )

    try:
        explainer = shap.KernelExplainer(lstm_predict_fn, X_background)
        print("KernelExplainer created successfully!")

        # Calculate SHAP values
        print("Calculating SHAP values...")
        print(
            "Note: This process may take several minutes due to KernelExplainer's sampling approach..."
        )

        shap_values = explainer.shap_values(X_explanation)

        print(f"SHAP values calculated successfully!")
        print(f"SHAP values shape: {shap_values.shape}")

        # Get expected value (baseline)
        expected_value = explainer.expected_value
        print(f"Expected value (baseline): {expected_value:.4f}")

    except Exception as e:
        print(f"KernelExplainer failed: {str(e)}")
        raise

    return explainer, shap_values, X_explanation, expected_value


def plot_lstm_shap_analysis(
    model,
    explainer,
    shap_values,
    X_explanation,
    explanation_data,
    X_cols,
    y_cols,
    expected_value,
    manager,
):
    """Generate comprehensive SHAP plots for LSTM using KernelExplainer results"""
    print("\n" + "=" * 60)
    print("GENERATING LSTM SHAP PLOTS (KERNELEXPLAINER)")
    print("=" * 60)

    # Rescale feature data back to original range for visualization
    X_explanation_df = pd.DataFrame(X_explanation, columns=X_cols)
    X_explanation_rescaled = rescale_data(X_explanation_df, X_cols)

    # Calculate scaling factor for SHAP values
    # SHAP values represent CONTRIBUTIONS (differences from baseline)
    scale_factor = (U_BOUND - L_BOUND) / 2  # Convert from [-1,1] scale to mg/dL scale
    shap_values_rescaled = shap_values * scale_factor

    # Rescale expected value
    expected_value_rescaled = expected_value * scale_factor + (U_BOUND + L_BOUND) / 2

    # Print SHAP values statistics for interpretation
    print(f"SHAP values statistics (in mg/dL contributions):")
    print(f"  Min SHAP value: {shap_values_rescaled.min():.2f} mg/dL")
    print(f"  Max SHAP value: {shap_values_rescaled.max():.2f} mg/dL")
    print(f"  Mean |SHAP| value: {np.abs(shap_values_rescaled).mean():.2f} mg/dL")
    print(f"  Expected value (baseline): {expected_value_rescaled:.2f} mg/dL")

    # Get model predictions using wrapper
    lstm_predict_fn = create_lstm_wrapper(model, X_cols, config["device"])
    predictions_scaled = lstm_predict_fn(X_explanation)

    # Rescale predictions
    predictions_rescaled = rescale_data(
        pd.DataFrame({"pred": predictions_scaled}), ["pred"]
    )["pred"].values

    # Rescale true values
    y_true_rescaled = rescale_data(
        pd.DataFrame({"target": explanation_data[y_cols[-1]].values}), ["target"]
    )["target"].values

    print(
        f"Predictions range: {predictions_rescaled.min():.2f} - {predictions_rescaled.max():.2f}"
    )
    print(
        f"True values range: {y_true_rescaled.min():.2f} - {y_true_rescaled.max():.2f}"
    )

    # 1. Summary plot (feature importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_rescaled,
        X_explanation_rescaled.values,
        feature_names=X_cols,
        show=False,
    )
    plt.title(
        "LSTM SHAP Summary Plot (KernelExplainer) - Feature Contributions to Glucose Prediction\n"
        "(SHAP values show mg/dL contribution of each feature)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"{config['shap_plots_dir']}/lstm_shap_kernel_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 2. Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_rescaled,
        X_explanation_rescaled.values,
        feature_names=X_cols,
        plot_type="bar",
        show=False,
    )
    plt.title(
        "LSTM SHAP Bar Plot (KernelExplainer) - Average Feature Importance",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        f"{config['shap_plots_dir']}/lstm_shap_kernel_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 3. Clinical case analysis - Waterfall plots for Hypo, Normal, Hyper cases
    print("Generating clinical case analysis for Hypo/Normal/Hyper glucose levels...")

    # Find representative cases for each glucose condition using shared sampling
    clinical_cases = find_clinical_cases(manager)

    # Generate waterfall plots for clinical cases
    generate_clinical_waterfall_plots(
        clinical_cases,
        shap_values_rescaled,
        X_explanation_rescaled,
        predictions_rescaled,
        y_true_rescaled,
        expected_value_rescaled,
        X_cols,
    )

    return predictions_rescaled


def save_shap_results(
    shap_values,
    X_explanation,
    explanation_data,
    predictions,
    X_cols,
):
    """Save SHAP analysis results to CSV"""
    print("\n" + "=" * 60)
    print("SAVING SHAP RESULTS")
    print("=" * 60)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "Patient_ID": explanation_data["Patient_ID"].values,
            "Timestamp": explanation_data["Timestamp"].values,
            "Prediction": predictions,
            "True_Value": rescale_data(
                pd.DataFrame({"target": explanation_data["lead30"].values}), ["target"]
            )["target"].values,
        }
    )

    # Add feature values
    for i, col in enumerate(X_cols):
        results_df[f"Feature_{col}"] = X_explanation[:, i]

    # Add SHAP values (already in mg/dL scale)
    scale_factor = (U_BOUND - L_BOUND) / 2
    shap_values_rescaled = shap_values * scale_factor

    for i, col in enumerate(X_cols):
        results_df[f"SHAP_{col}"] = shap_values_rescaled[:, i]

    # Save results
    results_path = f"{config['shap_plots_dir']}/lstm_shap_kernel_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"SHAP results saved to: {results_path}")

    return results_df


def print_summary_statistics(shap_values, predictions, y_true, X_cols):
    """Print summary statistics of the SHAP analysis"""
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS SUMMARY (KERNELEXPLAINER)")
    print("=" * 60)

    print(f"Number of samples analyzed: {len(predictions)}")
    print(f"Number of features: {len(X_cols)}")

    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction: {np.mean(predictions):.2f} mg/dL")
    print(f"  Std prediction: {np.std(predictions):.2f} mg/dL")
    print(f"  Min prediction: {np.min(predictions):.2f} mg/dL")
    print(f"  Max prediction: {np.max(predictions):.2f} mg/dL")

    # Scale SHAP values for statistics
    scale_factor = (U_BOUND - L_BOUND) / 2
    shap_values_rescaled = shap_values * scale_factor

    print(f"\nSHAP Values Statistics (Contributions in mg/dL):")
    print(f"  Mean |SHAP|: {np.mean(np.abs(shap_values_rescaled)):.4f} mg/dL")
    print(f"  Std |SHAP|: {np.std(np.abs(shap_values_rescaled)):.4f} mg/dL")
    print(f"  Max |SHAP|: {np.max(np.abs(shap_values_rescaled)):.4f} mg/dL")
    print(f"  Min SHAP: {np.min(shap_values_rescaled):.4f} mg/dL")
    print(f"  Max SHAP: {np.max(shap_values_rescaled):.4f} mg/dL")

    # Feature importance ranking
    mean_abs_shap = np.abs(shap_values_rescaled).mean(axis=0)
    feature_importance = list(zip(X_cols, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print(f"\nFeature Importance Ranking (by mean |SHAP| value):")
    for i, (feature, importance) in enumerate(feature_importance):
        print(f"  {i+1}. {feature}: {importance:.4f} mg/dL")

    # Calculate prediction accuracy
    mae = np.mean(np.abs(predictions - y_true))
    rmse = np.sqrt(np.mean((predictions - y_true) ** 2))

    print(f"\nModel Performance on Analyzed Samples:")
    print(f"  MAE: {mae:.2f} mg/dL")
    print(f"  RMSE: {rmse:.2f} mg/dL")


def main():
    """Main function to run LSTM SHAP analysis with KernelExplainer"""
    print("Starting LSTM SHAP Analysis with KernelExplainer...")

    # Print configuration
    print_config()

    # Setup directories
    setup_directories()

    # Set random seed
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    try:
        # Load model
        model = load_lstm_model()

        # Prepare data
        combined_data, background_data, explanation_data, X_cols, y_cols, manager = (
            prepare_shap_data()
        )

        # Perform SHAP analysis
        explainer, shap_values, X_explanation, expected_value = analyze_lstm_shap(
            model, background_data, explanation_data, X_cols
        )

        # Generate plots
        predictions = plot_lstm_shap_analysis(
            model,
            explainer,
            shap_values,
            X_explanation,
            explanation_data,
            X_cols,
            y_cols,
            expected_value,
            manager,
        )

        # Get true values for statistics
        y_true = rescale_data(
            pd.DataFrame({"target": explanation_data[y_cols[-1]].values}), ["target"]
        )["target"].values

        # Save results
        results_df = save_shap_results(
            shap_values,
            X_explanation,
            explanation_data,
            predictions,
            X_cols,
        )

        # Print summary
        print_summary_statistics(shap_values, predictions, y_true, X_cols)

        print("\n" + "=" * 60)
        print("LSTM SHAP ANALYSIS WITH KERNELEXPLAINER COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved in: {config['shap_plots_dir']}")
        print(f"Generated plots:")
        plots = [
            "lstm_shap_kernel_summary.png",
            "lstm_shap_kernel_bar.png",
            "lstm_shap_kernel_waterfall_hypo.png",
            "lstm_shap_kernel_waterfall_normal.png",
            "lstm_shap_kernel_waterfall_hyper.png",
        ]
        for plot in plots:
            print(f"  - {plot}")

        print(f"Generated CSV files:")
        csv_files = ["lstm_shap_kernel_results.csv"]
        for csv_file in csv_files:
            print(f"  - {csv_file}")

        print("\nNote: KernelExplainer provides model-agnostic explanations")
        print(
            "Results may differ slightly from gradient-based methods due to sampling approach"
        )

    except Exception as e:
        print(f"\nError during SHAP analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
