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

used_cols = [
    "tipo_vendedor",
    "quartos",
    "suites",
    "vagas",
    "area_util",
    "area_extra"
]

# %%
# Read from csv and turn into array
data = read_csv(TRAIN_FILE)

x = data[used_cols].to_numpy()
y = data[["preco"]].to_numpy().reshape((-1,))

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
            "rmse": results["mean_test_score"][i],
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

rs = RandomizedSearchCV(
    SVR(),
    dist,
    n_jobs=15,
    cv=cv,
    refit=True,
    scoring="neg_root_mean_squared_error",
    n_iter=50
)

rs.fit(X=x, y=y)

# Log results
log_results("svm", rs.cv_results_)
print(f"Best result: {rs.best_score_:.5f} and params {rs.best_params_}")

# %%
# Train SVM (with standard scalers)
dist = [{
    # "reg__C": np.linspace(1e7, 1e9, 150),
    "reg__C": uniform(loc=1e7, scale=1e7/4),
    "reg__gamma": uniform(loc=0.05, scale=0.05),
    # "reg__gamma": np.linspace(0, 1, 150),
}]

svm_scaler = StandardScaler()
model = Pipeline((
    ("std", svm_scaler),
    ("reg", SVR())
))

rs = RandomizedSearchCV(
    model,
    dist,
    n_jobs=15,
    cv=cv,
    refit=True,
    scoring="neg_root_mean_squared_error",
    n_iter=500
)

rs.fit(X=x, y=y)

# Log results
log_results("std-svm", rs.cv_results_)
print(f"Best result: {rs.best_score_:.5f} and params {rs.best_params_}")

# %%
# Train MPL
dist = [
    dict(
        reg__hidden_layer_sizes=[(5,2), (5,3), (5,4), (10,2), (10,3), (10,4)],
        reg__activation=['logistic', 'relu'],
        reg__solver=["lbfgs", "sgd", "adam"],
        reg__batch_size=np.linspace(10, 200, 6, dtype=np.int32),
        reg__learning_rate_init=uniform(loc=0.001, scale=0.002),
        reg__alpha=uniform(loc=0.001, scale=0.002)
    )
]

mlp_scaler = StandardScaler()
model = Pipeline((
    ("std", mlp_scaler),
    ("reg", MLPRegressor(max_iter=3000, early_stopping=True))
))

rs = RandomizedSearchCV(
    model,
    dist,
    n_jobs=15,
    cv=cv,
    refit=True,
    scoring="neg_root_mean_squared_error",
    n_iter=50
)

rs.fit(X=x, y=y)

# Log results
log_results("std-mlp", rs.cv_results_)
print(f"Best result: {rs.best_score_:.5f} and params {rs.best_params_}")

# %%
# Train final model
final_scaler = StandardScaler()
final_model = Pipeline((
    ("std", final_scaler),
    ("reg", SVR(
        C=12453751.825599143,
        gamma=0.05171257493033211
    ))
))

final_model.fit(x, y)

# %%
# Read test data and eval it
TEST_DATA = "test_data.csv"
test_data = read_csv(TEST_DATA)

# Exclude id for inference
test_x = test_data[used_cols].to_numpy()
test_y = final_model.predict(test_x)

test_data["preco"] = test_y
res = test_data[["Id", "preco"]]
res.to_csv("t2_results.csv", index=False)
