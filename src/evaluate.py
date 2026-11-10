import json
from pathlib import Path
import numpy as np
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

m = mlflow.sklearn.load_model("exported_model")
meta = json.loads(Path("exported_model/metadata.json").read_text())
threshold = meta.get("threshold_medinc")


d = fetch_california_housing(as_frame=True)
_, Xtest, _, ytest_reg = train_test_split(
    d.data, d.target, test_size=0.2, random_state=42
)
ytest = (ytest_reg > threshold).astype(int)

pred = m.predict(Xtest)
r = {
    "test_accuracy": accuracy_score(ytest, pred),
    "test_f1_macro": f1_score(ytest, pred, average="macro"),
}
Path("test_metrics.json").write_text(json.dumps(r, indent=2))
print(r)

