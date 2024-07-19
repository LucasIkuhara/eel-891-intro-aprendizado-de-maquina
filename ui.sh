#!/bin/sh
python -m mlflow ui -p 8080 --backend-store-uri $DB_CONN_STRING
