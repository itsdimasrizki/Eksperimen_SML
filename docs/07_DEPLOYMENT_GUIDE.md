# Deployment Guide

## Serving

mlflow models serve

atau

uvicorn app:app

## Endpoint

POST /predict

Request:
{
  "age": 60,
  "sex": 1,
  ...
}

Response:
{
  "prediction": 1
}