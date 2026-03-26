# MLflow: A "Hello World" Example

- Date: 2026-03-26
- Summary: A brief introduction to MLflow, an open-source platform for managing the machine learning lifecycle, with a basic Python "hello world" example demonstrating experiment tracking.

## Notes
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It was spearheaded by Matei Zaharia at Databricks. MLflow provides tools for experiment tracking, reproducible runs, model packaging, and model deployment.

Here's a simple "Hello World" Python example demonstrating MLflow's experiment tracking:

```python
import mlflow

if __name__ == "__main__":
mlflow.log_param("greeting", "Hello, MLflow!")
mlflow.log_metric("version", 1.0)
print("Logged 'Hello, MLflow!' as a parameter and version 1.0 as a metric.")
```

To run this example and view the results in the MLflow UI, you would typically:
1. Install MLflow: `pip install mlflow`
2. Save the code above as a Python file (e.g., `hello_mlflow.py`).
3. Run the script: `python hello_mlflow.py`
4. Start the MLflow UI: `mlflow ui`
5. Open your web browser to `http://localhost:5000` to see the logged experiment.

## Sources
- https://en.wikipedia.org/wiki/Databricks
- https://www.simplificando.tech/mlops/mlflow/hello_world_mlflow.html
