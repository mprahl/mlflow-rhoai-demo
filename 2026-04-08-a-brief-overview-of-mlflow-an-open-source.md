# MLflow Overview and Hello World

- Date: 2026-04-08
- Summary: A brief overview of MLflow, an open-source platform for managing the machine learning lifecycle, along with a simple Python "Hello World" example demonstrating basic experiment tracking.

## Notes
MLflow is an open-source platform designed to manage the end-to-end machine learning (ML) lifecycle. It provides a set of tools for experiment tracking, reproducible runs, model packaging, and model deployment. This helps data scientists and engineers to effectively track and compare experiments, package their code for reproducibility, and deploy ML models.

### Python Hello World Example with MLflow

This simple example demonstrates how to log a parameter and a metric using MLflow.

```python
import mlflow

# Log a parameter
mlflow.log_param("greeting", "Hello, MLflow!")

# Log a metric
mlflow.log_metric("version", 1.0)

print("MLflow Hello World experiment logged!")
```

To view the logged experiment, you would typically run `mlflow ui` in your terminal from the directory where your MLflow runs are stored.

## Sources
- https://www.simplificando.tech/mlops/mlflow/hello_world_mlflow.html
