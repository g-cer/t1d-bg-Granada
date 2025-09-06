"""
Script di esempio per recuperare parametri ottimali da studi Optuna salvati.

Questo script mostra come caricare gli studi Optuna salvati dal unified_tuning.py
e recuperare i parametri ottimali per uso futuro.
"""

import pickle
import pandas as pd
import os


def load_and_analyze_study(model_name, results_dir="tuning/results"):
    """Carica e analizza uno studio Optuna salvato."""
    study_path = f"{results_dir}/{model_name}_optuna_study.pkl"

    if not os.path.exists(study_path):
        print(f"❌ Studio non trovato: {study_path}")
        return None

    # Carica lo studio
    with open(study_path, "rb") as f:
        study = pickle.load(f)

    print(f"\n{'='*60}")
    print(f"ANALISI STUDIO OPTUNA - {model_name.upper()}")
    print(f"{'='*60}")

    # Informazioni di base
    print(f"Nome studio: {study.study_name}")
    print(f"Numero trials: {len(study.trials)}")
    print(f"Migliore MAE: {study.best_value:.4f}")
    print(f"Numero trial migliore: {study.best_trial.number}")

    # Parametri ottimali
    print(f"\nPARAMETRI OTTIMALI:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")

    # Statistiche sui trials
    trials_df = study.trials_dataframe()
    print(f"\nSTATISTICHE TRIALS:")
    print(f"  MAE medio: {trials_df['value'].mean():.4f}")
    print(f"  MAE deviazione standard: {trials_df['value'].std():.4f}")
    print(f"  MAE minimo: {trials_df['value'].min():.4f}")
    print(f"  MAE massimo: {trials_df['value'].max():.4f}")

    # Top 5 trials
    print(f"\nTOP 5 TRIALS:")
    top_trials = trials_df.nsmallest(5, "value")[["number", "value"]]
    for idx, row in top_trials.iterrows():
        print(f"  Trial {int(row['number'])}: MAE = {row['value']:.4f}")

    return study


def compare_models(results_dir="tuning/results"):
    """Confronta i risultati di tutti i modelli disponibili."""
    models = []

    for model_name in ["xgb", "lgb"]:
        study_path = f"{results_dir}/{model_name}_optuna_study.pkl"
        if os.path.exists(study_path):
            with open(study_path, "rb") as f:
                study = pickle.load(f)

            models.append(
                {
                    "Modello": model_name.upper(),
                    "Best MAE": f"{study.best_value:.4f}",
                    "Num Trials": len(study.trials),
                    "Range MAE": f"{min(t.value for t in study.trials):.4f} - {max(t.value for t in study.trials):.4f}",
                }
            )

    if models:
        print(f"\n{'='*80}")
        print("CONFRONTO MODELLI")
        print(f"{'='*80}")

        df = pd.DataFrame(models)
        print(df.to_string(index=False))

        # Trova il migliore
        best_model = min(models, key=lambda x: float(x["Best MAE"]))
        print(
            f"\n🏆 MODELLO MIGLIORE: {best_model['Modello']} con MAE = {best_model['Best MAE']}"
        )
    else:
        print("❌ Nessun studio Optuna trovato.")


def main():
    """Funzione principale di esempio."""
    print("ANALISI STUDI OPTUNA SALVATI")
    print("=" * 80)

    # Analizza tutti i modelli disponibili
    for model_name in ["xgb", "lgb"]:
        study = load_and_analyze_study(model_name)

    # Confronta tutti i modelli
    compare_models()

    print(f"\n{'='*80}")
    print("COME UTILIZZARE I PARAMETRI OTTIMALI:")
    print(f"{'='*80}")
    print(
        """
# Esempio per XGBoost:
import pickle
import xgboost as xgb

# Carica lo studio
with open('tuning/results/xgb_optuna_study.pkl', 'rb') as f:
    study = pickle.load(f)

# Ottieni parametri ottimali
best_params = study.best_params
print(f"Best MAE: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# Crea modello con parametri ottimali
# (ricorda di combinare con parametri base se necessario)
model = xgb.XGBRegressor(**combined_params)
model.fit(X_train, y_train)
"""
    )


if __name__ == "__main__":
    main()
