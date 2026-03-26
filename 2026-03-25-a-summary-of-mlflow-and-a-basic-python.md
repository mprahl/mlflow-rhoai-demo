# MLflow Overview with Hello World Example

- Date: 2026-03-25
- Summary: A summary of MLflow and a basic Python "Hello World" example demonstrating its use for logging parameters and metrics.

## Notes
MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides tools for experiment tracking, reproducible runs, model packaging, and model deployment. This allows data scientists and machine learning engineers to track experiments, share code, and deploy models efficiently.

Here's a simple "Hello World" example using Python and MLflow:

```python
import mlflow

# Log a parameter
mlflow.log_param("greeting", "Hello, MLflow!")

# Log a metric
mlflow.log_metric("version", 1.0)

# Print a message
print("MLflow Hello World example completed.")
```

## Sources
- https://www.simplificando.tech/mlops/mlflow/hello_world_mlflow.html
