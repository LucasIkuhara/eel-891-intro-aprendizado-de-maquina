# %%
# Imports
from mlflow import log_metrics, log_params, start_run, log_input
import mlflow
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import numpy as np
from scipy.stats import uniform

# %%
# Read training data
TRAIN_FILE = "train_ds.csv"
SPLITS = 5
EXP_ID = "2"
LOG_TO_DB = True

# %%
# Read from csv and turn into array
data = read_csv(TRAIN_FILE)

x = data.drop(columns=["preco"]).to_numpy()
y = data[["preco"]].to_numpy()

# Log the dataset used
dataset = mlflow.data.from_pandas(data, name="real-estate-train")

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
        log_input(dataset)


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
# Linear regressor (without standard scalers)
model = LinearRegression()
score = cross_val_score(model, x, y, cv=cv, scoring="neg_root_mean_squared_error")
print(f"Linear Regression results for baseline: RMSE={-score.mean():.1f} +- {score.std():.1f}")

# %%
# Train SVM (no scalers)
dist = [{
    "C": uniform(loc=2, scale=2),
    "gamma": np.linspace(0.001, 0.1, num=5)
}]

gs = RandomizedSearchCV(
    SVR(),
    dist,
    n_jobs=15,
    cv=cv,
    refit=True,
    scoring="neg_root_mean_squared_error",
    n_iter=50
)

gs.fit(X=x, y=y)

# Log results
log_results("svm", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
# Train SVM (with standard scalers)
param_grid = [{
    "clf__C": np.linspace(0.25, 0.45, num=40),
    "clf__gamma": np.linspace(0.01, 0.3, num=15)
}]

model = Pipeline((
    ("std", StandardScaler()),
    ("clf", SVC())
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
log_results("std-svm", gs.cv_results_)
print(f"Best result: {gs.best_score_:.5f} and params {gs.best_params_}")

# %%
# Train MPL
param_grid = [
    dict(
        clf__hidden_layer_sizes=[(10,2), (10,3), (10,4)],
        clf__activation=['logistic', 'relu'],
        clf__solver=["lbfgs", "sgd", "adam"],
        clf__batch_size=[10, 15, 20, 25],
        clf__learning_rate_init=[0.05, 0.001, 0.005],
        clf__alpha=[0.0001/2, 0.0001, 0.0001*2]
    )
]


model = Pipeline((
    ("std", StandardScaler()),
    ("clf", MLPClassifier(max_iter=1000, early_stopping=True))
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
# Train final model
final_model = Pipeline((
    ("std", StandardScaler()),
    ("clf", MLPClassifier(
        activation="relu",
        batch_size=15,
        alpha=5e-5,
        hidden_layer_sizes=(10, 4),
        solver="sgd",
        learning_rate_init=0.05
    ))
))

final_model.fit(x, y)

# %%
# Read test data and eval it
TEST_DATA = "clean_test_ds.csv"
test_data = read_csv(TEST_DATA)

# Reorder columns
col_list = data.columns[:COLS_USED]
test_data = test_data[["id_solicitante"] + list(col_list)]

# Exclude id for inference
test_x = test_data[test_data.columns[1:COLS_USED + 1]].to_numpy()
test_y = final_model.predict(test_x)

test_data["inadimplente"] = test_y
res = test_data[["id_solicitante", "inadimplente"]]
res.to_csv("results.csv", index=False)
