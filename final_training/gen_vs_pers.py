import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
from utils.data import load_splits, rescale_data, calculate_metrics
from utils.tf_dnn import (
    create_gru_model,
    predict_in_batches,
    print_model_summary,
    create_callbacks,
)
from utils.clarke_error_grid import clarke_error_grid_analysis


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gru_results_file", type=str, default="outputs/test_set/gru_output.csv"
    )
    parser.add_argument(
        "--xgb_results_file", type=str, default="outputs/test_set/xgb_output.csv"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/personalized")
    parser.add_argument("--models_dir", type=str, default="models/personalized")
    parser.add_argument("--scores_dir", type=str, default="scores/gen_vs_pers")
    parser.add_argument("--plots_dir", type=str, default="plots/gen_vs_pers")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epochs", type=int, default=100, help="Max epochs per GRU personalizzato"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size per GRU personalizzato"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate per GRU personalizzato"
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=20,
        help="Early stopping patience per GRU personalizzato",
    )

    return parser.parse_args()


def setup_environment(args):
    """Setup directories and random seeds"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.scores_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"Output directory: {args.output_dir}")


def set_seeds(seed):
    """Set random seeds for reproducibility in TensorFlow/Keras."""
    # Set NumPy seed
    np.random.seed(seed)
    # Set TensorFlow seeds
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def identify_extreme_patients(gru_results_file):
    """Identifica pazienti con migliore e peggiore performance dal GRU"""
    print("\n🔍 Identificando pazienti estremi dal test set GRU...")

    results_df = pd.read_csv(gru_results_file)

    # Calcola metriche per paziente
    patient_metrics = []
    for patient_id in results_df["Patient_ID"].unique():
        patient_data = results_df[results_df["Patient_ID"] == patient_id]
        samples, maes, mapes, rmses = calculate_metrics(patient_data)
        patient_metrics.append(
            {
                "Patient_ID": patient_id,
                "Samples": samples,
                "MAE": maes[0] if len(maes) > 0 else np.nan,
                "MAPE": mapes[0] if len(mapes) > 0 else np.nan,
                "RMSE": rmses[0] if len(rmses) > 0 else np.nan,
            }
        )

    patient_metrics_df = pd.DataFrame(patient_metrics).dropna()

    # Identifica estremi basati su MAE
    best_patient_id = patient_metrics_df.loc[
        patient_metrics_df["MAE"].idxmin(), "Patient_ID"
    ]
    worst_patient_id = patient_metrics_df.loc[
        patient_metrics_df["MAE"].idxmax(), "Patient_ID"
    ]

    print(f"✓ Migliore paziente: {best_patient_id}")
    print(f"✓ Peggiore paziente: {worst_patient_id}")

    return int(best_patient_id), int(worst_patient_id)


def load_original_data():
    """Carica i dati originali completi"""
    print("\n📁 Caricando dati originali...")

    # Carica splits originali per ottenere X_cols e y_cols
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    # Combina tutti i dati per avere il dataset completo
    all_data = pd.concat([train_set, val_set, test_set], ignore_index=True)
    all_data = all_data.sort_values(["Patient_ID", "Timestamp"]).reset_index(drop=True)

    print(f"✓ Caricati {len(all_data)} campioni totali")

    return all_data, X_cols, y_cols


def create_patient_splits(patient_data, test_split, patient_id):
    """Crea split temporali per un singolo paziente (80% train, 20% test)"""
    # Ordina per timestamp per mantenere ordine temporale
    patient_data = patient_data.sort_values("Timestamp").reset_index(drop=True)

    # Split temporale: primi 80% per train, ultimi 20% per test
    split_idx = int(len(patient_data) * (1 - test_split))

    train_data = patient_data.iloc[:split_idx].copy()
    test_data = patient_data.iloc[split_idx:].copy()

    print(f"  Paziente {patient_id}: {len(train_data)} train, {len(test_data)} test")

    return train_data, test_data


def train_personalized_xgb(train_data, X_cols, y_cols, patient_id, args):
    """Addestra XGBoost personalizzato per un paziente"""
    print(f"\n🌲 Addestrando XGBoost personalizzato per paziente {patient_id}...")

    # Carica parametri ottimizzati da Optuna
    optuna_study_path = "tuning/results/xgb_optuna_study.pkl"

    try:
        with open(optuna_study_path, "rb") as f:
            study = pickle.load(f)
        best_params = study.best_params

        # Usa i parametri ottimizzati
        print(f"  Usando parametri ottimizzati: {best_params}")

    except FileNotFoundError:
        print(f"  ⚠️ File {optuna_study_path} non trovato")

    model = xgb.XGBRegressor(
        **best_params,
        random_state=args.seed,
        device="cuda:0",
    )

    model.fit(
        train_data[X_cols],
        train_data[y_cols[-1]],
    )

    # Salva modello
    model_path = f"{args.models_dir}/xgb_patient_{patient_id}.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✓ Modello XGBoost personalizzato salvato: {model_path}")

    return model


def train_personalized_gru(train_data, test_data, X_cols, y_cols, patient_id, args):
    """Addestra GRU personalizzato per un paziente"""
    print(f"\n🧠 Addestrando GRU personalizzato per paziente {patient_id}...")

    # Prepara dati per GRU (usa il train-set per l'addestramento e il test-set per l'early stopping)
    X_train = train_data[X_cols].values.reshape(len(train_data), len(X_cols), 1)
    y_train = train_data[y_cols[-1]].values

    X_test = test_data[X_cols].values.reshape(len(test_data), len(X_cols), 1)
    y_test = test_data[y_cols[-1]].values

    print(f"  Forma dati training: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  Forma dati testing: X_test {X_test.shape}, y_test {y_test.shape}")

    # Crea modello GRU con architettura ottimale
    print(f"\nCreating personalized GRU model...")

    set_seeds(args.seed)

    model = create_gru_model()

    print_model_summary(model)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.Huber(),
        metrics=["mae"],
        steps_per_execution=256,
    )

    callbacks = create_callbacks(
        early_stopping_patience=args.es_patience, early_stopping_min_delta=0
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Salva i pesi del miglior modello alla fine del training
    # (EarlyStopping con restore_best_weights=True già ripristina i migliori pesi)
    model_save_path = f"{args.models_dir}/gru_patient_{patient_id}.weights.h5"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_weights(model_save_path)
    print(f"Best model weights saved to: {model_save_path}")

    return model


def evaluate_personalized_models(
    patient_id, test_data, X_cols, y_cols, xgb_model, gru_model, args
):
    """Valuta modelli personalizzati su dati test del paziente"""
    print(f"\n📊 Valutando modelli personalizzati per paziente {patient_id}...")

    # Predizioni XGBoost personalizzato
    xgb_pred = xgb_model.predict(test_data[X_cols])

    # Predizioni GRU personalizzato
    gru_pred = predict_in_batches(gru_model, test_data[X_cols], "gru")

    # Crea DataFrame risultati
    results = test_data.copy()
    results["target"] = test_data[y_cols[-1]]
    results["xgb_personalized"] = xgb_pred
    results["gru_personalized"] = gru_pred

    # Rescale ai valori originali
    results = rescale_data(results, ["target", "xgb_personalized", "gru_personalized"])

    # Salva risultati
    output_file = f"{args.output_dir}/xgb_and_gru_output_patient_{patient_id}.csv"
    results[
        [
            "Timestamp",
            "Patient_ID",
            "bgClass",
            "target",
            "xgb_personalized",
            "gru_personalized",
        ]
    ].to_csv(output_file, index=False)

    print(f"✓ Risultati personalizzati salvati: {output_file}")

    return results


def load_generalized_results(patient_id, xgb_results_file, gru_results_file):
    """Carica risultati modelli generalizzati per un paziente specifico"""
    print(f"\n📂 Caricando risultati generalizzati per paziente {patient_id}...")

    # Carica risultati XGBoost generalizzato
    xgb_results = pd.read_csv(xgb_results_file)
    xgb_patient = xgb_results[xgb_results["Patient_ID"] == patient_id].copy()

    # Carica risultati GRU generalizzato
    gru_results = pd.read_csv(gru_results_file)
    gru_patient = gru_results[gru_results["Patient_ID"] == patient_id].copy()

    # Crea dataset combinato
    generalized_results = xgb_patient[
        ["Timestamp", "Patient_ID", "bgClass", "target"]
    ].copy()
    generalized_results["xgb_generalized"] = xgb_patient["y_pred"].values
    generalized_results["gru_generalized"] = gru_patient["y_pred"].values

    print(f"✓ Caricati {len(generalized_results)} risultati generalizzati")

    return generalized_results


def calculate_comparison_metrics(
    patient_id, personalized_results, generalized_results, args
):
    """Calcola metriche comparative tra modelli generalizzati e personalizzati"""
    print(f"\n📈 Calcolando metriche comparative per paziente {patient_id}...")

    # Usa il dataset personalizzato come riferimento (ultimi 20% dei dati)
    # e trova i corrispondenti nel dataset generalizzato basandosi sull'ordine temporale

    print(f"  Personalizzato: {len(personalized_results)} campioni")
    print(f"  Generalizzato: {len(generalized_results)} campioni")

    # Prendi gli ultimi N campioni dal dataset generalizzato dove N = len(personalized_results)
    n_samples = len(personalized_results)
    generalized_subset = generalized_results.tail(n_samples).reset_index(drop=True)
    personalized_subset = personalized_results.reset_index(drop=True)

    # Usa target dal dataset personalizzato
    target = personalized_subset["target"].values

    print(f"  {len(target)} campioni per confronto")

    # Calcola metriche per tutti i modelli
    metrics_data = []

    models = {
        "XGBoost_Generalized": generalized_subset["xgb_generalized"].values,
        "XGBoost_Personalized": personalized_subset["xgb_personalized"].values,
        "GRU_Generalized": generalized_subset["gru_generalized"].values,
        "GRU_Personalized": personalized_subset["gru_personalized"].values,
    }

    for model_name, predictions in models.items():
        mae = mean_absolute_error(target, predictions)
        mape = mean_absolute_percentage_error(target, predictions)
        rmse = root_mean_squared_error(target, predictions)

        # Clarke Error Grid
        ceg_stats = clarke_error_grid_analysis(
            target,
            predictions,
            f"{model_name} - Patient {patient_id}",
            save_path=f"{args.plots_dir}/ceg_{model_name.lower()}_patient_{patient_id}.png",
        )

        metrics_data.append(
            {
                "Patient_ID": patient_id,
                "Model": model_name,
                "Samples": len(target),
                "MAE": mae,
                "MAPE": mape,
                "RMSE": rmse,
                "Zone_A_Pct": ceg_stats["zone_percentages"]["A"],
                "Zone_B_Pct": ceg_stats["zone_percentages"]["B"],
                "Zone_C_Pct": ceg_stats["zone_percentages"]["C"],
                "Zone_D_Pct": ceg_stats["zone_percentages"]["D"],
                "Zone_E_Pct": ceg_stats["zone_percentages"]["E"],
                "Clinically_Acceptable": ceg_stats["clinically_acceptable"],
                "Clinically_Dangerous": ceg_stats["clinically_dangerous"],
            }
        )

    return pd.DataFrame(metrics_data)


def create_comparison_summary(all_metrics_df, args):
    """Crea riassunto finale del confronto"""
    print("\n📋 Creando riassunto finale del confronto...")

    # Salva metriche complete
    metrics_file = f"{args.scores_dir}/complete_comparison.csv"
    all_metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Metriche complete salvate: {metrics_file}")

    # Crea tabella riassuntiva
    summary_data = []

    for patient_id in all_metrics_df["Patient_ID"].unique():
        patient_metrics = all_metrics_df[all_metrics_df["Patient_ID"] == patient_id]

        for metric in ["MAE", "MAPE", "RMSE", "Clinically_Acceptable"]:
            row = {"Patient_ID": patient_id, "Metric": metric}

            for model in [
                "XGBoost_Generalized",
                "XGBoost_Personalized",
                "GRU_Generalized",
                "GRU_Personalized",
            ]:
                value = patient_metrics[patient_metrics["Model"] == model][
                    metric
                ].values[0]
                if metric in ["MAE", "RMSE"]:
                    row[model] = f"{value:.2f}"
                elif metric == "MAPE":
                    row[model] = f"{value:.2f}%"
                else:  # Clinically_Acceptable
                    row[model] = f"{value:.2f}%"

            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    # summary_file = f"{output_dir}/comparison_summary.csv"
    # summary_df.to_csv(summary_file, index=False)
    # print(f"✓ Riassunto salvato: {summary_file}")

    # Stampa riassunto
    print("\n" + "=" * 80)
    print("CONFRONTO MODELLI GENERALIZZATI VS PERSONALIZZATI")
    print("=" * 80)

    for patient_id in sorted(all_metrics_df["Patient_ID"].unique()):
        print(f"\n🔍 PAZIENTE {patient_id}:")
        patient_data = summary_df[summary_df["Patient_ID"] == patient_id]

        for _, row in patient_data.iterrows():
            print(
                f"  {row['Metric']:<20}: XGB_Gen {row['XGBoost_Generalized']:<8} | XGB_Pers {row['XGBoost_Personalized']:<8} | GRU_Gen {row['GRU_Generalized']:<8} | GRU_Pers {row['GRU_Personalized']:<8}"
            )

    print("\n" + "=" * 80)

    return summary_df


def main():
    """Funzione principale"""
    args = parse_arguments()

    print("🚀 CONFRONTO MODELLI GENERALIZZATI VS PERSONALIZZATI")
    print("=" * 60)

    # Setup environment
    setup_environment(args)

    # Identifica pazienti estremi
    best_patient, worst_patient = identify_extreme_patients(args.gru_results_file)
    target_patients = [best_patient, worst_patient]

    # Carica dati originali
    all_data, X_cols, y_cols = load_original_data()

    # Raccoglitore per tutte le metriche
    all_metrics = []

    # Processa ogni paziente target
    for patient_id in target_patients:
        print(f"\n" + "=" * 50)
        print(f"PROCESSING PAZIENTE {patient_id}")
        print("=" * 50)

        # Estrai dati del paziente
        patient_data = all_data[all_data["Patient_ID"] == patient_id].copy()
        print(f"Dati totali paziente {patient_id}: {len(patient_data)} campioni")

        # Crea split temporali per addestramento modelli personalizzati
        train_data, test_data = create_patient_splits(
            patient_data, args.test_split, patient_id
        )

        # Addestra modelli personalizzati
        xgb_personalized = train_personalized_xgb(
            train_data, X_cols, y_cols, patient_id, args
        )
        gru_personalized = train_personalized_gru(
            train_data, test_data, X_cols, y_cols, patient_id, args
        )

        # Valuta modelli personalizzati
        personalized_results = evaluate_personalized_models(
            patient_id,
            test_data,
            X_cols,
            y_cols,
            xgb_personalized,
            gru_personalized,
            args,
        )

        # Carica risultati modelli generalizzati
        generalized_results = load_generalized_results(
            patient_id, args.xgb_results_file, args.gru_results_file
        )

        # Calcola metriche comparative
        patient_metrics = calculate_comparison_metrics(
            patient_id, personalized_results, generalized_results, args
        )

        if patient_metrics is not None:
            all_metrics.append(patient_metrics)

    # Combina tutte le metriche
    if all_metrics:
        all_metrics_df = pd.concat(all_metrics, ignore_index=True)

        # Crea riassunto finale
        summary_df = create_comparison_summary(all_metrics_df, args)

        print(f"\n✅ Confronto completato! Risultati salvati in: {args.output_dir}")

        return {
            "metrics": all_metrics_df,
            "summary": summary_df,
            "target_patients": target_patients,
        }
    else:
        print("\n❌ Nessun risultato generato")
        return None


if __name__ == "__main__":
    results = main()
