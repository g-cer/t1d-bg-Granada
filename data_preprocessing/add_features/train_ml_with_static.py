import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data import load_splits
from training.ml.train_std_ml import create_model, save_model, evaluate_and_save_results


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/val_set")
    parser.add_argument("--models_path", type=str, default="models/val_set")
    parser.add_argument("--plots_path", type=str, default="plots/val_set")
    parser.add_argument(
        "--exp_name",
        type=str,
        choices=["lgb", "xgb", "rf"],
        # required=True,
        default="xgb",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def analyze_feature_importance(model, X_cols, plots_dir):
    """Analyze and visualize feature importance."""
    print("📈 Analyzing feature importance...")

    # Get feature importance
    importance_gain = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": X_cols, "importance": importance_gain}
    ).sort_values("importance", ascending=False)

    # Categorize features
    def categorize_feature(feature):
        if feature.startswith("lag"):
            return "Glucose Lag"
        elif feature in ["Sex", "Age"]:
            return "Demographics"
        elif feature in ["HbA1c", "TSH", "Creatinine", "HDL", "Triglycerides"]:
            return "Biochemical"
        else:
            return "Other"

    importance_df["category"] = importance_df["feature"].apply(categorize_feature)

    # Print top features
    print("\n🏆 Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(
            f"   {row['feature']:<15} ({row['category']:<12}): {row['importance']:.4f}"
        )

    # Summary by category
    category_importance = importance_df.groupby("category")["importance"].agg(
        ["sum", "mean", "count"]
    )
    print(f"\n📊 Feature Importance by Category:")
    for category, stats in category_importance.iterrows():
        print(
            f"   {category:<12}: Total={stats['sum']:.4f}, Mean={stats['mean']:.4f}, Count={stats['count']}"
        )

    return importance_df, category_importance


def create_feature_importance_plots(importance_df, category_importance, plots_dir):
    """Create and save feature importance visualizations."""
    print("🎨 Creating feature importance visualizations...")

    os.makedirs(plots_dir, exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create a single figure for the feature importance plot
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # 1. Top 15 Individual Features
    top_features = importance_df.head(15)
    colors = [
        (
            "#1f77b4"
            if cat == "Glucose Lag"
            else "#ff7f0e" if cat == "Biochemical" else "#2ca02c"
        )
        for cat in top_features["category"]
    ]

    bars1 = ax1.barh(range(len(top_features)), top_features["importance"], color=colors)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features["feature"])
    ax1.set_xlabel("Feature Importance (Gain)")
    ax1.set_title("Top 15 Most Important Features", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars1, top_features["importance"])):
        ax1.text(
            val + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    # Save plot
    plot_path = f"{plots_dir}/xgb_static_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✓ Feature importance plot saved to: {plot_path}")

    # Create a summary table plot
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Create biochemical features comparison
    biochemical_features = importance_df[
        importance_df["category"] == "Biochemical"
    ].copy()
    if not biochemical_features.empty:
        biochemical_features = biochemical_features.sort_values(
            "importance", ascending=True
        )

        bars = ax.barh(
            range(len(biochemical_features)),
            biochemical_features["importance"],
            color="#ff7f0e",
            alpha=0.7,
        )
        ax.set_yticks(range(len(biochemical_features)))
        ax.set_yticklabels(biochemical_features["feature"])
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title(
            "Biochemical Features Importance Ranking", fontsize=16, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars, biochemical_features["importance"])):
            ax.text(
                val + max(biochemical_features["importance"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontweight="bold",
            )

    plt.tight_layout()

    # Save biochemical plot
    biochemical_plot_path = f"{plots_dir}/xgb_biochemical_features_importance.png"
    plt.savefig(biochemical_plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✓ Biochemical features plot saved to: {biochemical_plot_path}")


def main():
    """Main training function"""
    args = parse_arguments()

    print(f"Starting {args.exp_name.upper()} training pipeline...")

    # Setup
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.models_path, exist_ok=True)
    os.makedirs(args.plots_path, exist_ok=True)

    # Load data
    print("Loading pre-prepared static data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits("data/static_split_sets")

    # Create and train model
    print(f"Creating and training {args.exp_name.upper()} model...")
    model = create_model(args.exp_name, args.seed)
    model.fit(train_set[X_cols], train_set[y_cols[-1]])

    # Save model
    model_path = f"{args.models_path}/{args.exp_name}_static.pickle"
    save_model(model, model_path)

    print()  # Add spacing

    # Evaluate and save results
    results = evaluate_and_save_results(
        model, val_set, X_cols, y_cols, args.output_path, args.exp_name + "_static"
    )

    # Analyze feature importance
    importance_df, category_importance = analyze_feature_importance(
        model, X_cols, args.plots_path
    )

    # Create visualizations
    create_feature_importance_plots(importance_df, category_importance, args.plots_path)

    print(f"\n🎉 XGBoost training with static features completed successfully!")
    print(f"\n📊 Key Insights:")
    glucose_importance = (
        category_importance.loc["Glucose Lag", "sum"]
        if "Glucose Lag" in category_importance.index
        else 0
    )
    biochemical_importance = (
        category_importance.loc["Biochemical", "sum"]
        if "Biochemical" in category_importance.index
        else 0
    )
    total_importance = category_importance["sum"].sum()

    print(
        f"   • Glucose lag features contribute {glucose_importance/total_importance*100:.1f}% of total importance"
    )
    print(
        f"   • Biochemical features contribute {biochemical_importance/total_importance*100:.1f}% of total importance"
    )
    print(
        f"   • Top biochemical feature: {importance_df[importance_df['category']=='Biochemical'].iloc[0]['feature'] if not importance_df[importance_df['category']=='Biochemical'].empty else 'None'}"
    )

    return model, results, importance_df


if __name__ == "__main__":
    model, results, importance_df = main()
