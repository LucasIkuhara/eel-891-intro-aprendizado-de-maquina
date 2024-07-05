# %%
# Imports
from mlflow import log_metrics, log_params, start_run
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# %%
# Read training data
INPUT_FILE = "clean_training_ds.csv"
OUTPUT_FILE = "selected_features.csv"
SPLITS = 5
EXP_ID = "1"

# Read from csv and turn into array
data = read_csv(INPUT_FILE)
data = data.dropna()
y = data["inadimplente"].to_numpy()
features = data.drop(["inadimplente"], axis=1)
x = data.to_numpy()

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
            "technique": "logistic-regression"
        }
    ):
        log_params(params)
        log_metrics(metrics)

# %%
# Compute feature importance
# Using all columns, fit and then get Feat. imp.
model = RandomForestClassifier()
pipe = Pipeline((
    ("std", StandardScaler()),
    ("clf", model)
))
pipe.fit(x, y)
importance = sorted(
    zip(
        model.feature_importances_,
        features.columns
    ), 
    key=lambda x: x[0], 
    reverse=True
)

print(importance)

# %%
# Test with increasing number of x columns used.
best = (0, 0, [])
for i in range(1, len(data.columns) + 1):

    # Get the names of the first i columns based on feature importance
    columns = [name[1] for name in importance[:i]]
    curr_x = features[columns].to_numpy()

    params = {
        "C": 0.3,
        "penalty": "l1",
        "solver": "liblinear",
        "max_iter": 10_000
    }

    model = LogisticRegression(**params)
    pipe = Pipeline((
        ("std", StandardScaler()),
        ("clf", model)
    ))
    score = cross_val_score(pipe, curr_x, y, cv=SPLITS, n_jobs=SPLITS)
    acc = score.mean()
    metrics = {
        "accuracy": acc,
        "std": score.std()
    }

    if acc > best[0]:
        best = (acc, i, columns)

    # log_results("column-selection", params, metrics, i)
    print(f"{i} columns used, accuracy: {acc}")

print(f"Best result is {best[0]:.3f} with {best[1]} columns. ({best[2]})")

# %%
# Dump selected features
out_cols = best[2] + ["inadimplente"]
data[out_cols].to_csv(OUTPUT_FILE, index=False)
