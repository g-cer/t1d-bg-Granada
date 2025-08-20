import os
import argparse
from utils_data import *
import pickle
import lightgbm as lgb
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


class CFG:
    l_bound = 40.0
    u_bound = 400.0
    train_split = list(map(int, np.load("data/patients/train_patients.npy")))
    test_split = list(map(int, np.load("data/patients/test_patients.npy")))
    horizons = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
    output_csv_header = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data", type=str)
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument("--cache_dir", type=str, default="training/data_cache")
parser.add_argument("--use_cache", action="store_true", default=True)
parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild cache")
parser.add_argument(
    "--exp_name",
    type=str,
    choices=["lgb", "xgb", "fast_rf"],
    help="Model name. Choose from [lgb, xgb, fast_rf]",
    required=True,
)
parser.add_argument("--seed", type=int, default=42)


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Use cache unless force rebuild is requested
    use_cache = args.use_cache and not args.force_rebuild

    train_set, test_set, X_cols, y_cols = get_data(
        args.data_path,
        horizons=CFG.horizons,
        train_split=CFG.train_split,
        test_split=CFG.test_split,
        use_cache=use_cache,
        cache_dir=args.cache_dir,
        scale=True,
    )

    # Train model
    if args.exp_name == "fast_rf":
        base = DecisionTreeRegressor(
            max_depth=16, min_samples_leaf=10, random_state=args.seed
        )
        model = BaggingRegressor(
            estimator=base,
            n_estimators=200,
            max_samples=200_000,
            bootstrap=True,
            n_jobs=-1,
            random_state=args.seed,
        )
    elif args.exp_name == "lgb":
        model = lgb.LGBMRegressor(random_state=args.seed)
    elif args.exp_name == "xgb":
        model = xgb.XGBRegressor(random_state=args.seed)

    model.fit(train_set[X_cols], train_set[y_cols[-1]])
    with open(f"{args.output_path}/{args.exp_name}.pickle", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n")

    # Evaluate
    test_set["y_pred"] = model.predict(test_set[X_cols])
    test_set = test_set.rename(columns={y_cols[-1]: "target"})
    test_set = rescale_data(test_set, ["target", "y_pred"])
    test_set = test_set[CFG.output_csv_header]

    print_results(test_set)

    test_set.to_csv(f"{args.output_path}/{args.exp_name}_output.csv", index=False)
