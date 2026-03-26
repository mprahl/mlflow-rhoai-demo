# MLflow Overview and Python Quickstart

- Date: 2026-03-26
- Summary: MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. This note includes a basic Python quickstart example for MLflow Tracking.

## Notes
# MLflow Overview

MLflow is an open-source platform designed to manage the end-to-end machine learning (ML) lifecycle. It addresses key challenges in ML development, including:
- **Tracking experiments**: Recording and comparing parameters, metrics, and artifacts.
- **Reproducibility**: Packaging ML code in a reusable and reproducible way.
- **Model deployment**: Deploying ML models to various serving platforms.

MLflow consists of several components:
- **MLflow Tracking**: An API and UI for logging parameters, code versions, metrics, and output files when running ML code and for later visualizing the results.
- **MLflow Projects**: A standard format for packaging reusable ML code.
- **MLflow Models**: A convention for packaging ML models in multiple formats to allow them to be used in a variety of downstream tools.
- **MLflow Model Registry**: A centralized model store to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations.

# Python Quickstart Example for MLflow Tracking

This example demonstrates how to use MLflow Tracking to log parameters, metrics, and a model.

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080") # Uncomment and modify if you have a tracking server

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():
    # Load the Iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

    # Define hyperparameters
params = {"solver": "liblinear", "random_state": 42}

    # Train a Logistic Regression model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
accuracy = lr.score(X_test, y_test)

    # Log the hyperparameters
mlflow.log_params(params)

    # Log the loss metric
mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
model_info = mlflow.sklearn.log_model(
sk_model=lr,
artifact_path="iris_model",
signature=signature,
input_example=X_train[:2]
)

print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
print(f"Model saved in: {model_info.artifact_path}")
```

## Sources
- https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html
