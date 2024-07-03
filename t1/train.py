# %%
# Imports
from mlflow import log_metrics, log_params, start_run
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from pandas import read_csv
from sklearn.linear_model import LogisticRegression

# %%
# Read training data
DATA_FILE = "clean_training_ds.csv"
SPLITS = 10
EXP_ID = "1"
COLS_USED = 17


# Read from csv and turn into array
data = read_csv(DATA_FILE)
data = data.dropna()
y = data["inadimplente"].to_numpy()
data = data.drop(["inadimplente"], axis=1)
x = data.to_numpy()
x = x.T[:COLS_USED].T

# Shuffle with set random state for reproducibility
cv = KFold(SPLITS, shuffle=True, random_state=1000)

# %%
# Define logging presets
def log_from_grid(name_prefix: str, params, metrics):

    # Append params to name
    name = name_prefix
    for p in params:
        params["columns-used"] = COLS_USED

        # Format floats with 2 decimals
        val = params[p]
        if isinstance(val, float): val = f"{val:.2f}"
        name += f"-{p}-{params[p]}"

    print(f"Logging model: {name}...")
    with start_run(
        experiment_id=EXP_ID,
        run_name=name,
        tags={
            "technique": name_prefix,
            "objective": "acc-optimization",
        }
    ):
        log_params(params)
        log_metrics(metrics)

# %%
# Train Logistic regressor
param_grid = [
    {"C": [0.1*(i+1) for i in range(20)], "penalty": ["l2", None]},
    {"C": [0.1*(i+1) for i in range(20)], "penalty": ["l1"], "solver": ["liblinear"]}
]

gs = GridSearchCV(
    LogisticRegression(),
    n_jobs=15,
    param_grid=param_grid,
    cv=cv
)
gs.fit(X=x, y=y)
results = gs.cv_results_

# Log results
for i in range(len(results["mean_test_score"])):

    params = results["params"][i]
    metrics = {
        "accuracy": results["mean_test_score"][i],
        "fit-time": results["mean_fit_time"][i]
    }

    log_from_grid("logistic-regression", params=params, metrics=metrics)
