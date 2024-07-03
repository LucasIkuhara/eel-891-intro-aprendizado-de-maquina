#!/bin/sh
python -m mlflow ui --backend-store-uri $(grep -oP '^MLFLOW_TRACKING_URI=.*$' .env | cut -d= -f2-) -p 8080
