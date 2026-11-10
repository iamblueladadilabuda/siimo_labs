import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
c = MlflowClient()


v = c.get_model_version_by_alias("housing-classifier", "champion")


model = mlflow.sklearn.load_model(f"models:/housing-classifier/{v.version}")


out = Path("exported_model")
if out.exists():
    shutil.rmtree(out)
mlflow.sklearn.save_model(
    model, str(out),
    pip_requirements=Path("requirements.txt").read_text().splitlines()
)


d = fetch_california_housing(as_frame=True)
X_full, y_reg = d.data, d.target


Xdev, Xtest, ydev_reg, ytest_reg = train_test_split(
    X_full, y_reg, test_size=0.2, random_state=42
)

Xtrain, _, ytrain_reg, _ = train_test_split(
    Xdev, ydev_reg, test_size=0.25, random_state=42
)


threshold = float(np.median(ytrain_reg))


metadata = {
    "name": "housing-classifier",
    "version": v.version,
    "run_id": v.run_id,
    "features": d.data.columns.tolist(),
    "classes": ["below_median", "above_median"],
    "threshold_medinc": threshold,
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2))


request = {"instances": Xtest.iloc[:3].to_dict(orient="records")}
Path("sample_request.json").write_text(json.dumps(request, indent=2))
Path("expected_predictions.json").write_text(
    json.dumps(model.predict(Xtest.iloc[:3]).tolist())
)

print(json.dumps(metadata, indent=2))

