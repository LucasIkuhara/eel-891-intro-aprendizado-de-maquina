import mlflow

mlflow.autolog()
mlflow.start_run(experiment_id="credit-default-classifier")


# Finish run logging
mlflow.end_run()
