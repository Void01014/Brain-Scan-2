"""Import the required frameworks and exercise TensorFlow, PyTorch, API and MLflow."""
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tmp" / "ultralytics"))

def main():
    versions = {}
    for name in ("tensorflow", "torch", "ultralytics", "mlflow", "fastapi", "streamlit",
                 "numpy", "pandas", "sklearn", "matplotlib", "seaborn", "PIL", "yaml", "ipykernel"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "imported")
        print(f"{name}: {versions[name]}", flush=True)
    import tensorflow as tf
    import torch
    from fastapi.testclient import TestClient
    from api.main import app
    import mlflow

    assert float(tf.reduce_sum(tf.constant([1.0, 2.0])).numpy()) == 3.0
    assert torch.tensor([1, 2]).sum().item() == 3
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "models_ready": False}
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("environment-smoke-test")
    with mlflow.start_run(run_name="setup-verification") as run:
        mlflow.log_param("purpose", "environment_setup")
        mlflow.log_metric("smoke_test_passed", 1.0)
        mlflow.log_dict(versions, "environment/versions.json")
        run_id = run.info.run_id
    saved = mlflow.MlflowClient().get_run(run_id)
    assert saved.data.metrics["smoke_test_passed"] == 1.0
    assert mlflow.MlflowClient().list_artifacts(run_id, "environment")
    report = {"python": sys.version, "versions": versions,
              "tensorflow_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
              "torch_cuda": torch.cuda.is_available(), "mlflow_run_id": run_id}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS: framework imports, tensor operations, API health, MLflow parameter/metric/artifact")

if __name__ == "__main__":
    main()
