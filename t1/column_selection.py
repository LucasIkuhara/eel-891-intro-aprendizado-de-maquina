# %%
# Imports
from mlflow import log_metrics, log_params, start_run
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from sklearn.linear_model import LogisticRegression

# %%
# Read training data
DATA_FILE = "clean_training_ds.csv"
SPLITS = 10
EXP_ID = "1"

# Read from csv and turn into array
data = read_csv(DATA_FILE)
data = data.dropna()
y = data["inadimplente"].to_numpy()
data = data.drop(["inadimplente"], axis=1)
x = data.to_numpy()
column_count = x.shape[1]

# %%
# Define logging presets
def log_results(name_prefix: str, params, metrics, cols_used: int):

    # Append params to name
    name = f"{name_prefix}-cols-{cols_used}"
    params["columns-used"] = cols_used

    print(f"Logging model: {name}...")
    with start_run(
        experiment_id=EXP_ID,
        run_name=name,
        tags={
            "objective": "column-selection",
            "techinique": "logistic-regression"
        }
    ):
        log_params(params)
        log_metrics(metrics)

# %%
# Test with increasing number of x columns used.
for i in range(1, column_count + 1):
    curr_x = x.T[:i].T

    params = {
        'penalty': 'l1',
        "solver": 'liblinear',
    }

    model = LogisticRegression(**params)
    score = cross_val_score(model, curr_x, y, cv=SPLITS)
    metrics = {
        "accuracy": score.mean(),
        "std": score.std()
    }

    log_results("column-selection", params, metrics, i)
    print(f"{i} columns used, accuracy: {score.mean()}")
