import argparse, os
import mlflow
from mlflow import MlflowClient

p = argparse.ArgumentParser()
p.add_argument("--version", required=True)
p.add_argument("--alias", default="champion")
a = p.parse_args()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
c = MlflowClient()
c.set_registered_model_alias("housing-classifier", a.alias, a.version)
v = c.get_model_version("housing-classifier", a.version)
print(f"{a.alias} -> version {v.version}, run {v.run_id}")

