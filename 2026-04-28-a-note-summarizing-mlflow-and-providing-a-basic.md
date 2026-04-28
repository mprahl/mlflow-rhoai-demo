# MLflow Hello World Example

- Date: 2026-04-28
- Summary: A note summarizing MLflow and providing a basic Python 'hello world' example for tracking experiments.

## Notes
MLflow is an open-source platform developed by Databricks for managing the end-to-end machine learning lifecycle. It helps with experiment tracking, reproducible runs, and model packaging.

Here's a simple Python "hello world" example demonstrating basic MLflow tracking:

```python
import os
import mlflow
from random import random

def main():
mlflow.log_param("hello_param", "world")
mlflow.log_metric("hello_metric", random())
os.system(f"echo 'hello world' > helloworld.txt")
mlflow.log_artifact("helloworld.txt")

if __name__ == "__main__":
main()
```

This example shows how to:
- Log a parameter (`hello_param`).
- Log a metric (`hello_metric`).
- Log an artifact (a text file named `helloworld.txt`).

To run this, you would typically have an MLflow tracking server set up, and then execute the Python script. The logged parameters, metrics, and artifacts would be recorded in your MLflow experiment.

## Sources
- https://en.wikipedia.org/wiki/Databricks
- https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-cli-runs?view=azureml-api-2
