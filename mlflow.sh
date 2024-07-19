#!/bin/sh
mlflow server --host 127.0.0.1 --port $SERVER_PORT --backend-store-uri $DB_CONN_STRING &
mlflow ui -p $UI_PORT --backend-store-uri $DB_CONN_STRING
