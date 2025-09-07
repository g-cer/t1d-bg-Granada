import pandas as pd


def process_patient_info():
    """Process patient information file."""
    print("Processing patient_info...")

    df_patients = pd.read_csv("data/T1DiabetesGranada/Patient_info.csv")

    # Extract patient ID as integer
    df_patients["Patient_ID"] = df_patients["Patient_ID"].str[-4:].astype(int)

    # Calculate age at initial measurement date
    df_patients["Initial_measurement_date"] = pd.to_datetime(
        df_patients["Initial_measurement_date"]
    )
    df_patients["Age"] = (
        df_patients["Initial_measurement_date"].dt.year - df_patients["Birth_year"]
    )

    # Encode sex as binary
    sex_mapping = {"F": 0, "M": 1}
    df_patients["Sex"] = df_patients["Sex"].map(sex_mapping)

    # Select and save relevant columns
    df_patients = df_patients[["Patient_ID", "Sex", "Age"]]
    df_patients.to_csv("data/T1DiabetesGranada/Patient_info_corrected.csv", index=False)
    print("✓ Patient info file processed")


def process_biochemical_parameters():
    """Process biochemical parameters file."""
    print("Processing biochemical parameters...")

    df_biochem = pd.read_csv("data/T1DiabetesGranada/Biochemical_parameters.csv")

    # Extract patient ID as integer
    df_biochem["Patient_ID"] = df_biochem["Patient_ID"].str[-4:].astype(int)

    # Transform from long to wide format
    df_biochem_wide = df_biochem.pivot_table(
        index=["Patient_ID", "Reception_date"],
        columns="Name",
        values="Value",
        aggfunc="first",  # Handle potential duplicates
    ).reset_index()

    df_biochem_wide.columns.name = None

    # Rename columns to more readable names
    df_biochem = df_biochem_wide.rename(
        columns={
            "Reception_date": "Timestamp",
            "Glycated hemoglobin (A1c)": "HbA1c",
            "Thyrotropin (TSH)": "TSH",
            "Creatinine": "Creatinine",
            "HDL cholesterol": "HDL",
            "Triglycerides": "Triglycerides",
        }
    )

    # Select relevant columns
    df_biochem = df_biochem[
        [
            "Patient_ID",
            "Timestamp",
            "HbA1c",
            "TSH",
            "Creatinine",
            "HDL",
            "Triglycerides",
        ]
    ]

    df_biochem.to_csv(
        "data/T1DiabetesGranada/Biochemical_parameters_corrected.csv", index=False
    )
    print("✓ Biochemical parameters file processed")


if __name__ == "__main__":
    process_patient_info()
    process_biochemical_parameters()
