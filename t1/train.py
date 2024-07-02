import mlflow
from sklearn.model_selection import cross_val_score
from pandas import read_csv
from sklearn.linear_model import LogisticRegression


DATA_FILE = "clean_training_ds.csv"

mlflow.autolog()
mlflow.start_run(experiment_id=1)

use_cols_except = [
    "forma_envio_solicitacao",
    "estado_onde_nasceu",
    "estado_onde_reside",
    "possui_telefone_residencial",
    "codigo_area_telefone_residencial",
    "vinculo_formal_com_empresa",
    "estado_onde_trabalha",
    "possui_telefone_trabalho",
    "codigo_area_telefone_trabalho",
]

data = read_csv(DATA_FILE)
data = data.dropna()
y = data["inadimplente"].to_numpy()
data = data.drop(["inadimplente"] + use_cols_except, axis=1)
x = data.to_numpy()

model = LogisticRegression()
cv_score = cross_val_score(model, x, y, cv=8)

print(f"Scores: {cv_score} \n\nTotal: {cv_score.mean():.2f} +- {cv_score.std():.2f}")
# Finish run logging
mlflow.end_run()
