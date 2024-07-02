#!/bin/sh
python -m mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI -p 8080
