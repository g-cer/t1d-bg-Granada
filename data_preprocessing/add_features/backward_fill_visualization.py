import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np


def create_simplified_example():
    """Create a simplified visualization with synthetic but realistic data patterns."""

    # Read real patient data to get realistic patterns
    df_biochem = pd.read_csv(
        "data/T1DiabetesGranada/Biochemical_parameters_corrected.csv"
    )
    df_biochem["Timestamp"] = pd.to_datetime(df_biochem["Timestamp"])

    # Get a real patient with biochemical data
    patient_with_data = df_biochem[df_biochem["HbA1c"].notna()].head(1)
    if patient_with_data.empty:
        print("No patient with HbA1c data found")
        return

    real_patient_id = patient_with_data.iloc[0]["Patient_ID"]
    print(
        f"Creating visualization based on real patient patterns from Patient {real_patient_id}"
    )

    # Create realistic synthetic data for clear visualization
    base_date = datetime(2020, 1, 1)

    # Biochemical measurements (sparse, like real data)
    biochem_dates = [
        base_date + timedelta(days=0),  # Jan 1
        base_date + timedelta(days=90),  # Apr 1
        base_date + timedelta(days=180),  # Jul 1
        base_date + timedelta(days=270),  # Sep 27
    ]
    biochem_values = [7.2, 6.8, 6.5, 7.1]  # HbA1c values

    # Generate realistic glucose time series with 15-minute intervals
    glucose_dates = []
    glucose_values = []

    # Define the overall time range (1 year)
    start_date = base_date - timedelta(days=30)
    end_date = base_date + timedelta(days=365)

    # Generate continuous glucose measurements every 15 minutes
    current_time = start_date
    base_glucose = 120  # Base glucose level
    daily_pattern_amplitude = 30  # Daily glucose variation
    noise_level = 15  # Random noise

    # Simulate realistic daily glucose patterns
    while current_time <= end_date:
        # Create daily pattern (higher in morning/evening, lower at night)
        hour_of_day = current_time.hour + current_time.minute / 60
        daily_factor = (
            np.sin((hour_of_day - 6) * np.pi / 12) * 0.5 + 0.5
        )  # Peak around noon

        # Add some weekly variation
        day_of_week = current_time.weekday()
        weekly_factor = 1 + 0.1 * np.sin(day_of_week * np.pi / 3.5)

        # Calculate glucose value with realistic patterns
        glucose_value = (
            base_glucose
            + daily_factor * daily_pattern_amplitude
            + np.random.normal(0, noise_level) * weekly_factor
        )

        # Ensure realistic glucose range (70-250 mg/dL)
        glucose_value = max(70, min(250, glucose_value))

        glucose_dates.append(current_time)
        glucose_values.append(glucose_value)

        # Move to next 15-minute interval
        current_time += timedelta(minutes=15)

    # Create DataFrames
    df_biochem_viz = pd.DataFrame(
        {
            "Timestamp": biochem_dates,
            "HbA1c": biochem_values,
            "Patient_ID": [real_patient_id] * len(biochem_dates),
        }
    )

    df_glucose_viz = pd.DataFrame(
        {
            "Timestamp": glucose_dates,
            "Measurement": glucose_values,
            "Patient_ID": [real_patient_id] * len(glucose_dates),
        }
    )

    # Sort by timestamp
    df_glucose_viz = df_glucose_viz.sort_values("Timestamp").reset_index(drop=True)
    df_biochem_viz = df_biochem_viz.sort_values("Timestamp").reset_index(drop=True)

    # Perform backward merge
    merged_data = pd.merge_asof(
        df_glucose_viz,
        df_biochem_viz[["Timestamp", "HbA1c"]],
        on="Timestamp",
        direction="backward",
        tolerance=pd.Timedelta(days=30),
    )

    # Create visualization - Single combined plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    fig.suptitle(
        f"Processo di Backward Fill - Finestra di Tolleranza di 30 Giorni\n"
        + f"Time Series Glucosio (15 min) + Parametri Biochimici (Patient {real_patient_id})",
        fontsize=18,
        fontweight="bold",
    )

    # Plot glucose time series as a continuous line
    ax.plot(
        df_glucose_viz["Timestamp"],
        df_glucose_viz["Measurement"],
        color="steelblue",
        alpha=0.7,
        linewidth=1,
        label="Time Series Glucosio (15 min)",
        zorder=1,
    )

    # Highlight successful and failed matches
    successful_matches = merged_data.dropna()
    failed_matches = merged_data[merged_data["HbA1c"].isna()]

    # Plot successful matches as green points
    if not successful_matches.empty:
        ax.plot(
            successful_matches["Timestamp"],
            successful_matches["Measurement"],
            color="darkblue",
            alpha=0.8,
            # s=20,
            label=f"Match con HbA1c ({len(successful_matches)} punti)",
            zorder=3,
        )

    # # Plot failed matches as red points
    # if not failed_matches.empty:
    #     ax.scatter(
    #         failed_matches["Timestamp"],
    #         failed_matches["Measurement"],
    #         color="red",
    #         alpha=0.6,
    #         s=15,
    #         label=f"Senza match ({len(failed_matches)} punti)",
    #         zorder=2,
    #     )

    # Create secondary y-axis for HbA1c
    ax2 = ax.twinx()

    # Plot biochemical measurements with 30-day forward tolerance windows
    for idx, row in df_biochem_viz.iterrows():
        # Draw the tolerance window
        start_window = row["Timestamp"]
        end_window = row["Timestamp"] + pd.Timedelta(days=30)

        # Fill the tolerance window
        ax.axvspan(
            start_window,
            end_window,
            alpha=0.1,
            color="orange",
            zorder=0,
            label="Finestra 30 giorni" if idx == 0 else "",
        )

        # Draw vertical line at biochemical measurement
        ax.axvline(
            x=row["Timestamp"],
            color="darkred",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            zorder=4,
        )

    # Plot HbA1c measurements on secondary axis
    ax2.scatter(
        df_biochem_viz["Timestamp"],
        df_biochem_viz["HbA1c"],
        color="orangered",
        alpha=1.0,
        s=200,
        marker="s",
        edgecolors="darkred",
        linewidth=2,
        label="Misurazioni HbA1c",
        zorder=5,
    )

    # Add annotations for HbA1c values
    for idx, row in df_biochem_viz.iterrows():
        ax2.annotate(
            f'HbA1c: {row["HbA1c"]:.1f}%\n{row["Timestamp"].strftime("%Y-%m-%d")}',
            xy=(row["Timestamp"], row["HbA1c"]),
            xytext=(10, 20),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            zorder=6,
        )

    # Set labels and formatting
    ax.set_xlabel("Data", fontsize=14, fontweight="bold")
    ax.set_ylabel("Glucosio (mg/dL)", fontsize=14, fontweight="bold", color="darkblue")
    ax2.set_ylabel("HbA1c (%)", fontsize=14, fontweight="bold", color="firebrick")

    # Set y-axis limits
    ax.set_ylim(60, 250)
    ax2.set_ylim(6, 8)

    # Color the y-axis labels
    ax.tick_params(axis="y", labelcolor="darkblue")
    ax2.tick_params(axis="y", labelcolor="firebrick")

    # Grid and legend
    ax.grid(True, alpha=0.3)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Add detailed explanation text box
    total_glucose = len(df_glucose_viz)
    successful_merges = len(successful_matches)
    failed_merges = len(failed_matches)
    merge_rate = (successful_merges / total_glucose) * 100

    # Save the plot
    output_path = "plots/backward_fill_combined_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✅ Visualizzazione combinata salvata come: {output_path}")

    return merged_data, total_glucose, successful_merges, failed_merges


if __name__ == "__main__":
    print("🔄 CREAZIONE VISUALIZZAZIONE DETTAGLIATA DEL BACKWARD FILL")
    print("=" * 60)

    try:
        result = create_simplified_example()
        if result:
            merged_data, total_glucose, successful_merges, failed_merges = result

            print("\n✅ Visualizzazione combinata completata!")
            print("Controlla il file 'backward_fill_combined_visualization.png'")
            print(f"\nStatistiche:")
            print(f"- Misurazioni glucosio totali: {total_glucose:,} (ogni 15 minuti)")
            print(
                f"- Match riusciti: {successful_merges:,} ({successful_merges/total_glucose*100:.1f}%)"
            )
            print(
                f"- Match falliti: {failed_merges:,} ({failed_merges/total_glucose*100:.1f}%)"
            )
        else:
            print("❌ Errore nella creazione della visualizzazione")

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback

        traceback.print_exc()
