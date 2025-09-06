import os
import argparse
import pandas as pd
from utils.clarke_error_grid import clarke_error_grid_analysis


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/test_set")
    parser.add_argument("--plots_path", type=str, default="plots/test_set")
    parser.add_argument("--scores_path", type=str, default="scores/test_set")
    return parser.parse_args()


def extract_model_name_from_filename(filename):
    """Extract clean model name from output filename"""
    # Remove '_output.csv' suffix and convert to uppercase
    if filename.endswith("_output.csv"):
        model_name = filename[:-11].upper()
    else:
        model_name = filename.split(".")[0].upper()

    return model_name


def generate_clarke_plots_and_stats(output_path, plots_path, scores_path):
    """Generate Clarke Error Grid plots and statistics for all model outputs"""
    print("=" * 80)
    print("CLARKE ERROR GRID ANALYSIS")
    print("=" * 80)

    # Create output directories
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(scores_path, exist_ok=True)

    # Find all output files
    output_files = [f for f in os.listdir(output_path) if f.endswith("_output.csv")]

    if not output_files:
        print(f"❌ No output files found in {output_path}")
        print("   Looking for files ending with '_output.csv'")
        return

    print(f"Found {len(output_files)} model output files:")
    for file in output_files:
        print(f"  - {file}")

    # Store results for summary table
    ceg_results = []

    # Process each model output file
    for file in output_files:
        print(f"\n{'='*60}")
        model_name = extract_model_name_from_filename(file)
        print(f"PROCESSING: {model_name}")
        print(f"{'='*60}")

        # Load model results
        file_path = os.path.join(output_path, file)
        try:
            model_results = pd.read_csv(file_path)
            print(f"✓ Loaded {len(model_results)} predictions from {file}")
        except Exception as e:
            print(f"❌ Error loading {file}: {str(e)}")
            continue

        # Validate required columns
        required_cols = ["target", "y_pred"]
        missing_cols = [
            col for col in required_cols if col not in model_results.columns
        ]
        if missing_cols:
            print(f"❌ Missing required columns in {file}: {missing_cols}")
            continue

        # Extract reference and predicted values
        ref_values = model_results["target"].values
        pred_values = model_results["y_pred"].values

        # Generate Clarke Error Grid plot
        plot_filename = f"ceg_{model_name.lower()}.png"
        plot_path = os.path.join(plots_path, plot_filename)

        # Perform Clarke Error Grid analysis
        try:
            ceg_stats = clarke_error_grid_analysis(
                ref_values=ref_values,
                pred_values=pred_values,
                title_string=f"{model_name} Test Set",
                save_path=plot_path,
            )

            print(f"✓ Clarke Error Grid plot saved: {plot_filename}")

            # Print zone statistics
            print(f"\nClarke Error Grid Zone Statistics:")
            for zone, percentage in ceg_stats["zone_percentages"].items():
                print(f"  Zone {zone}: {percentage:.2f}%")

            print(
                f"  Clinically Acceptable (A+B): {ceg_stats['clinically_acceptable']:.2f}%"
            )
            print(
                f"  Clinically Dangerous (D+E): {ceg_stats['clinically_dangerous']:.2f}%"
            )

            # Store results for summary table
            ceg_results.append(
                {
                    "Model": model_name,
                    "Total_Points": ceg_stats["total_points"],
                    "Zone_A_Pct": f"{ceg_stats['zone_percentages']['A']:.2f}%",
                    "Zone_B_Pct": f"{ceg_stats['zone_percentages']['B']:.2f}%",
                    "Zone_C_Pct": f"{ceg_stats['zone_percentages']['C']:.2f}%",
                    "Zone_D_Pct": f"{ceg_stats['zone_percentages']['D']:.2f}%",
                    "Zone_E_Pct": f"{ceg_stats['zone_percentages']['E']:.2f}%",
                    "Clinically_Acceptable_AandB": f"{ceg_stats['clinically_acceptable']:.2f}%",
                    "Clinically_Dangerous_DandE": f"{ceg_stats['clinically_dangerous']:.2f}%",
                }
            )

        except Exception as e:
            print(f"❌ Error generating Clarke Error Grid for {model_name}: {str(e)}")
            continue

    # Save summary table
    if ceg_results:
        print(f"\n{'='*80}")
        print("SAVING CLARKE ERROR GRID SUMMARY")
        print(f"{'='*80}")

        # Create summary DataFrame
        ceg_df = pd.DataFrame(ceg_results)

        # Save to CSV
        summary_path = os.path.join(scores_path, "ceg_zones_results.csv")
        ceg_df.to_csv(summary_path, index=False)
        print(f"✓ CEG zone statistics saved to: {summary_path}")

        # Display summary table
        print(f"\nClarke Error Grid Zone Summary:")
        print(ceg_df.to_string(index=False))

        # Find best performing model (highest A+B percentage)
        best_model_idx = (
            ceg_df["Clinically_Acceptable_AandB"].str.rstrip("%").astype(float).idxmax()
        )
        best_model = ceg_df.iloc[best_model_idx]
        print(
            f"\n🏆 Best Clinical Accuracy: {best_model['Model']} "
            f"({best_model['Clinically_Acceptable_AandB']} in zones A+B)"
        )

    else:
        print("❌ No valid Clarke Error Grid results generated")

    print(f"\n{'='*80}")
    print("CLARKE ERROR GRID ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Plots saved to: {plots_path}")
    print(f"📈 Statistics saved to: {scores_path}")


def main():
    """Main function"""
    print("CLARKE ERROR GRID ANALYSIS FOR MODEL OUTPUTS")
    print("=" * 80)

    # Parse arguments
    args = parse_arguments()

    print(f"Configuration:")
    print(f"  Output path: {args.output_path}")
    print(f"  Plots path: {args.plots_path}")
    print(f"  Scores path: {args.scores_path}")

    # Check if output directory exists
    if not os.path.exists(args.output_path):
        print(f"❌ Output directory not found: {args.output_path}")
        return

    # Generate Clarke Error Grid plots and statistics
    generate_clarke_plots_and_stats(
        output_path=args.output_path,
        plots_path=args.plots_path,
        scores_path=args.scores_path,
    )


if __name__ == "__main__":
    main()
