# Data-Science Sample

A sample data-science project with an emphasis in machine-learning. It contains two separate different examples problems:
 - t1: A credit approval classification problem
 - t2: A real state pricing regression problem

## Experiment Tracking

For tracking metrics and artifacts, mlflow is used. For its datastore, Postgres and S3 are used. For that reason, the following environment variables must be set:
 - MLFLOW_TRACKING_URI
 