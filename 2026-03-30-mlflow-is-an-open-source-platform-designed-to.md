# MLflow: Streamlining the Machine Learning Lifecycle

- Date: 2026-03-30
- Summary: MLflow is an open-source platform designed to streamline the machine learning lifecycle, offering tools for tracking experiments, packaging code into reproducible runs, and managing models.

## Notes
MLflow is an open-source platform that simplifies the machine learning lifecycle, from experimentation to deployment. It provides a set of tools to manage various aspects of ML development, including:

*   **MLflow Tracking:** Records and queries experiments, including code, data, configuration, and results.
*   **MLflow Projects:** Packages ML code in a reusable and reproducible format.
*   **MLflow Models:** Manages and deploys ML models from various ML libraries to diverse serving platforms.
*   **MLflow Model Registry:** Provides a centralized model store to collaboratively manage the full lifecycle of an MLflow Model.

### Simple MLflow "Hello World" Example

This example demonstrates a basic MLflow run, logging a parameter and a metric.

```python
import mlflow

# Start an MLflow run
with mlflow.start_run():
    # Log a parameter
mlflow.log_param("greeting", "Hello, MLflow!")

    # Log a metric (a simple value)
mlflow.log_metric("random_number", 0.789)

print("MLflow 'Hello World' run completed.")
```

To view the results of this run, you would typically navigate to the directory where your script was executed and run `mlflow ui` in your terminal, then open your web browser to `http://localhost:5000`.

## Sources
- https://medium.com/@anna_ml_llm/a-comprehensive-guide-to-mlflow-what-it-is-its-pros-and-cons-and-how-to-use-it-in-your-python-468af13468c6
- https://blog.devgenius.io/mlflow-an-extended-hello-world-99739b68bf29
