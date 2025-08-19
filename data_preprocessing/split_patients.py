import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import matplotlib.pyplot as plt


def categorize_glucose(value):
    """
    Categorizza i valori di glucosio secondo i range specificati:
    - Hypoglycemia: < 70 mg/dL
    - Normoglycemia: 70-180 mg/dL
    - Hyperglycemia: > 180 mg/dL
    """
    if pd.isna(value):
        return "missing"
    elif value < 70:
        return "hypoglycemia"
    elif value <= 180:
        return "normoglycemia"
    else:
        return "hyperglycemia"


def calculate_patient_distribution(df):
    """
    Calcola la distribuzione delle categorie per ogni paziente
    """
    patient_stats = {}

    for i, patient_id in enumerate(df["Patient_ID"].unique()):
        if i % 10 == 0:
            print(f"Processing patient {i+1}/{df['Patient_ID'].nunique()}...")
        patient_data = df[df["Patient_ID"] == patient_id]["Measurement"].dropna()

        if len(patient_data) == 0:
            continue

        categories = patient_data.apply(categorize_glucose)

        total_measurements = len(patient_data)
        hypoglycemia_count = (categories == "hypoglycemia").sum()
        normoglycemia_count = (categories == "normoglycemia").sum()
        hyperglycemia_count = (categories == "hyperglycemia").sum()

        patient_stats[patient_id] = {
            "total_measurements": total_measurements,
            "hypoglycemia_count": hypoglycemia_count,
            "normoglycemia_count": normoglycemia_count,
            "hyperglycemia_count": hyperglycemia_count,
            "hypoglycemia_pct": hypoglycemia_count / total_measurements,
            "normoglycemia_pct": normoglycemia_count / total_measurements,
            "hyperglycemia_pct": hyperglycemia_count / total_measurements,
        }

    return patient_stats


def stratified_patient_split(df, test_size=0.2, random_state=42):
    """
    Splitta i pazienti mantenendo le proporzioni delle categorie glicemiche
    """
    # Calcola le statistiche per ogni paziente
    patient_stats = calculate_patient_distribution(df)

    # Crea un DataFrame con le statistiche dei pazienti
    stats_df = pd.DataFrame.from_dict(patient_stats, orient="index")
    stats_df["patient_id"] = stats_df.index

    # Calcola le proporzioni globali del dataset
    total_measurements = stats_df["total_measurements"].sum()
    global_hypo_pct = stats_df["hypoglycemia_count"].sum() / total_measurements
    global_normo_pct = stats_df["normoglycemia_count"].sum() / total_measurements
    global_hyper_pct = stats_df["hyperglycemia_count"].sum() / total_measurements

    print(f"Proporzioni globali del dataset:")
    print(f"Hypoglycemia: {global_hypo_pct:.3f} ({global_hypo_pct*100:.1f}%)")
    print(f"Normoglycemia: {global_normo_pct:.3f} ({global_normo_pct*100:.1f}%)")
    print(f"Hyperglycemia: {global_hyper_pct:.3f} ({global_hyper_pct*100:.1f}%)")
    print()

    # Crea bins per la stratificazione basati sulle proporzioni di ogni categoria
    # Questo approccio raggruppa i pazienti con proporzioni simili
    n_bins = 5

    stats_df["hypo_bin"] = pd.cut(
        stats_df["hypoglycemia_pct"], bins=n_bins, labels=False
    )
    stats_df["normo_bin"] = pd.cut(
        stats_df["normoglycemia_pct"], bins=n_bins, labels=False
    )
    stats_df["hyper_bin"] = pd.cut(
        stats_df["hyperglycemia_pct"], bins=n_bins, labels=False
    )

    # Crea una chiave di stratificazione combinata
    stats_df["strat_key"] = (
        stats_df["hypo_bin"].astype(str)
        + "_"
        + stats_df["normo_bin"].astype(str)
        + "_"
        + stats_df["hyper_bin"].astype(str)
    )

    # Filtra i gruppi che hanno almeno 2 pazienti per permettere lo split
    valid_groups = stats_df.groupby("strat_key").filter(lambda x: len(x) >= 2)

    if len(valid_groups) < len(stats_df):
        print(
            f"Attenzione: {len(stats_df) - len(valid_groups)} pazienti rimossi perché in gruppi troppo piccoli"
        )

    # Esegui lo split stratificato
    try:
        train_patients, test_patients = train_test_split(
            valid_groups["patient_id"].values,
            test_size=test_size,
            stratify=valid_groups["strat_key"].values,
            random_state=random_state,
        )
    except ValueError:
        # Fallback: split semplice se la stratificazione fallisce
        print("Stratificazione fallita, uso split semplice...")
        train_patients, test_patients = train_test_split(
            stats_df["patient_id"].values,
            test_size=test_size,
            random_state=random_state,
        )

    return train_patients, test_patients, patient_stats


def validate_split(df, train_patients, test_patients, patient_stats):
    """
    Valida la qualità dello split confrontando le proporzioni
    """
    # Calcola le proporzioni per il training set
    train_total = sum(patient_stats[p]["total_measurements"] for p in train_patients)
    train_hypo = sum(patient_stats[p]["hypoglycemia_count"] for p in train_patients)
    train_normo = sum(patient_stats[p]["normoglycemia_count"] for p in train_patients)
    train_hyper = sum(patient_stats[p]["hyperglycemia_count"] for p in train_patients)

    # Calcola le proporzioni per il test set
    test_total = sum(patient_stats[p]["total_measurements"] for p in test_patients)
    test_hypo = sum(patient_stats[p]["hypoglycemia_count"] for p in test_patients)
    test_normo = sum(patient_stats[p]["normoglycemia_count"] for p in test_patients)
    test_hyper = sum(patient_stats[p]["hyperglycemia_count"] for p in test_patients)

    # Proporzioni globali
    global_total = train_total + test_total
    global_hypo_pct = (train_hypo + test_hypo) / global_total
    global_normo_pct = (train_normo + test_normo) / global_total
    global_hyper_pct = (train_hyper + test_hyper) / global_total

    # Proporzioni training
    train_hypo_pct = train_hypo / train_total
    train_normo_pct = train_normo / train_total
    train_hyper_pct = train_hyper / train_total

    # Proporzioni test
    test_hypo_pct = test_hypo / test_total
    test_normo_pct = test_normo / test_total
    test_hyper_pct = test_hyper / test_total

    print("Confronto delle proporzioni:")
    print(
        f"{'Categoria':<15} {'Globale':<10} {'Training':<10} {'Test':<10} {'Diff Train':<12} {'Diff Test':<12}"
    )
    print("-" * 75)
    print(
        f"{'Hypoglycemia':<15} {global_hypo_pct:.3f}     {train_hypo_pct:.3f}     {test_hypo_pct:.3f}     {abs(global_hypo_pct-train_hypo_pct):.3f}        {abs(global_hypo_pct-test_hypo_pct):.3f}"
    )
    print(
        f"{'Normoglycemia':<15} {global_normo_pct:.3f}     {train_normo_pct:.3f}     {test_normo_pct:.3f}     {abs(global_normo_pct-train_normo_pct):.3f}        {abs(global_normo_pct-test_normo_pct):.3f}"
    )
    print(
        f"{'Hyperglycemia':<15} {global_hyper_pct:.3f}     {train_hyper_pct:.3f}     {test_hyper_pct:.3f}     {abs(global_hyper_pct-train_hyper_pct):.3f}        {abs(global_hyper_pct-test_hyper_pct):.3f}"
    )
    print()
    print(
        f"Pazienti nel training set: {len(train_patients)} ({len(train_patients)/(len(train_patients)+len(test_patients)):.1%})"
    )
    print(
        f"Pazienti nel test set: {len(test_patients)} ({len(test_patients)/(len(train_patients)+len(test_patients)):.1%})"
    )
    print(f"Misurazioni nel training set: {train_total}")
    print(f"Misurazioni nel test set: {test_total}")


def save_splits(df, train_patients, test_patients, output_dir="data/"):
    """
    Salva il dataset aggiornato senza i pazienti esclusi e gli array numpy degli ID pazienti
    """
    # Combina tutti i pazienti inclusi nello split
    all_included_patients = np.concatenate([train_patients, test_patients])

    # Filtra il dataset per includere solo i pazienti che sono stati splittati
    df_filtered = df[df["Patient_ID"].isin(all_included_patients)]

    # Salva il dataset aggiornato
    updated_path = f"{output_dir}Glucose_measurements_corrected.csv"
    df_filtered.to_csv(updated_path, index=False)

    # Salva gli array numpy degli ID pazienti
    train_patients_array = np.array(train_patients)
    test_patients_array = np.array(test_patients)

    patients_dir = os.path.join(output_dir, "patients")
    os.makedirs(patients_dir, exist_ok=True)

    np.save(f"{output_dir}patients/train_patients.npy", train_patients_array)
    np.save(f"{output_dir}patients/test_patients.npy", test_patients_array)

    print(f"Dataset aggiornato salvato: {updated_path}")
    print(f"Array numpy pazienti training: {output_dir}train_patients_ids.npy")
    print(f"Array numpy pazienti test: {output_dir}test_patients_ids.npy")
    print()
    print(f"Statistiche del dataset aggiornato:")
    print(f"- Pazienti training: {len(train_patients_array)}")
    print(f"- Pazienti test: {len(test_patients_array)}")
    print(f"- Totale righe dataset aggiornato: {len(df_filtered)}")


def main():
    # Carica il dataset
    df = pd.read_csv("data/Glucose_measurements_corrected.csv")

    print(
        f"Dataset caricato: {len(df)} righe, {df['Patient_ID'].nunique()} pazienti unici"
    )
    print()

    # Esegui lo split
    train_patients, test_patients, patient_stats = stratified_patient_split(
        df, test_size=0.2, random_state=42
    )

    # Valida lo split
    validate_split(df, train_patients, test_patients, patient_stats)

    # Salva i risultati
    save_splits(df, train_patients, test_patients, output_dir="data/")

    return train_patients, test_patients, patient_stats


if __name__ == "__main__":
    train_patients, test_patients, patient_stats = main()
