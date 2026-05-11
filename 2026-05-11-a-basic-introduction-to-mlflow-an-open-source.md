# MLflow Hello World Example

- Date: 2026-05-11
- Summary: A basic introduction to MLflow, an open-source platform for managing the end-to-end machine learning lifecycle, demonstrated with a simple Python "Hello World" example for experiment tracking.

## Notes
# MLflow Hello World Example

MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It provides a set of tools to track experiments, package code into reproducible runs, and share and deploy models.

This note demonstrates a simple "Hello World" example using MLflow for basic experiment tracking.

## Python Example

To run this example, you would typically have MLflow installed (`pip install mlflow`) and then execute the Python script. You can then view the tracked experiments by running `mlflow ui` in your terminal and navigating to `http://localhost:5000`.

```python
import mlflow

if __name__ == "__main__":
    # Start an MLflow run
with mlflow.start_run():
        # Log a parameter
mlflow.log_param("greeting", "Hello")

        # Log a metric
mlflow.log_metric("version", 1.0)

print("MLflow 'Hello World' experiment logged!")
print("Run 'mlflow ui' in your terminal to view the experiment.")
```

## Sources
- https://www.simplificando.tech/mlops/mlflow/hello_world_mlflow.html
- https://github.com/amesar/mlflow-examples/blob/master/python/hello_world/hello_world.py
- https://mlflow.org/docs/latest/ml/tutorials-and-examples/
