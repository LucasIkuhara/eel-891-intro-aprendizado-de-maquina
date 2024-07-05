# %%
# Imports
from mlflow import log_metrics, log_params, start_run
from sklearn.model_selection import KFold, GridSearchCV
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np

# %%
# Read training data
DATA_FILE = "selected_features.csv"
SPLITS = 5
EXP_ID = "1"
COLS_USED = 10
LOG_TO_DB = False


# Read from csv and turn into array
data = read_csv(DATA_FILE)
data = data.dropna()
y = data["inadimplente"].to_numpy()
features = data.drop(["inadimplente"], axis=1)
x = features[features.columns[:COLS_USED]].to_numpy()


# Shuffle with set random state for reproducibility
cv = KFold(SPLITS, shuffle=True, random_state=1000)

# %%
# Define logging presets
def log_from_grid(name_prefix: str, params, metrics):

    # Append params to name
    name = name_prefix
    for p in params:

        # Format floats with 2 decimals
        val = params[p]
        if isinstance(val, float): val = f"{val:.2f}"
        name += f"-{p}-{val}"

    params["columns-used"] = COLS_USED

    if not LOG_TO_DB: 
        return

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


def log_results(name, results):
    for i in range(len(results["mean_test_score"])):

        params = results["params"][i]
        metrics = {
            "accuracy": results["mean_test_score"][i],
            "fit-time": results["mean_fit_time"][i]
        }
        print(f'{name}: {results["mean_test_score"][i]:.5f}')
        log_from_grid(name, params=params, metrics=metrics)

# %%
# Train Logistic regressor (no scalers)
param_grid = [
    {"penalty": [None], "max_iter": [10000], "solver": ["lbfgs"]},
    {"C": np.linspace(0.1, 2, num=30), "penalty": ["l1"], "max_iter": [10000], "solver": ["liblinear"]},
    {"C": np.linspace(0.1, 2, num=30), "penalty": ["l2"], "max_iter": [10000], "solver": ["lbfgs"]}
]

gs = GridSearchCV(
    LogisticRegression(),
    n_jobs=8,
    param_grid=param_grid,
    cv=cv,
    refit=True
)

gs.fit(X=x, y=y)

# Log results
log_results("logistic-regression", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
# Train Logistic regressor (with standard scalers)
param_grid = [
    {"clf__penalty": [None], "clf__solver": ["lbfgs"]},
    {"clf__C": np.linspace(0.1, 2, num=30), "clf__penalty": ["l1"], "clf__solver": ["liblinear"]},
    {"clf__C": np.linspace(0.1, 2, num=30), "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
]

model = Pipeline((
    ("std", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
))

gs = GridSearchCV(
    model,
    n_jobs=8,
    param_grid=param_grid,
    cv=cv,
    refit=True
)

gs.fit(X=x, y=y)

# Log results
log_results("std-logistic-regression", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
# Train SVM (no scalers)
param_grid = [{
    "C": np.linspace(0.01, 0.2, num=5),
    "gamma": np.linspace(0.01, 0.2, num=5)
}]

gs = GridSearchCV(
    SVC(),
    n_jobs=8,
    param_grid=param_grid,
    cv=cv,
    refit=True
)

gs.fit(X=x, y=y)

# Log results
log_results("svm", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
# Train SVM (with standard scalers)
param_grid = [{
    "clf__C": np.linspace(0.4, 0.6, num=10),
    "clf__gamma": np.linspace(0.025, 0.035, num=5)
}]

model = Pipeline((
    ("std", StandardScaler()),
    ("clf", SVC())
))

gs = GridSearchCV(
    model,
    n_jobs=15,
    param_grid=param_grid,
    cv=cv,
    refit=True
)

gs.fit(X=x, y=y)

# Log results
log_results("std-svm", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
# Train MPL
from sklearn.neural_network import MLPClassifier

param_grid = [{
    "clf__hidden_layer_sizes": [(1, 10), (2, 10), (3, 10), (4, 10)]
}]

model = Pipeline((
    ("std", StandardScaler()),
    ("clf", MLPClassifier())
))

gs = GridSearchCV(
    model,
    n_jobs=15,
    param_grid=param_grid,
    cv=cv,
    refit=True
)

gs.fit(X=x, y=y)

# Log results
log_results("std-mlp", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
