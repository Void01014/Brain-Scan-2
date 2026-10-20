# BrainScan 2

Brain MRI classification (TensorFlow/Keras GoogLeNet/InceptionV1), tumor detection
(YOLOv8), MLflow tracking, FastAPI, Streamlit, and Airflow orchestration.
The original assignment is saved in [docs/brief-original.txt](docs/brief-original.txt).

## Windows setup

Use Python 3.12 (64-bit). In PowerShell from this folder:

```powershell
.\scripts\setup.ps1 -Python 'C:\path\to\python.exe'
```

This creates `.venv`, installs `requirements-lock.txt` when available (otherwise
`requirements.txt`), creates `.env`, and runs the environment smoke check.
No activation is required; use `.venv\Scripts\python.exe` directly.
For notebooks, select this interpreter in your editor.
The setup uses a local `.tmp` directory for temporary installation files.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_env.py
```

The smoke check imports the frameworks, runs TensorFlow and PyTorch tensor
operations, checks the API health route, and records and reads back an MLflow
parameter, metric, and JSON artifact. Its report is `reports/environment.json`.
The Windows lock file is platform-specific; use `requirements.txt` for Linux.

## Run the services

Run each command in a separate terminal in the project folder:

Shortcuts: `.\scripts\start.ps1 api`, `.\scripts\start.ps1 ui`, or
`.\scripts\start.ps1 mlflow`.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app/main.py --server.address 127.0.0.1
.\.venv\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

- API documentation: http://127.0.0.1:8000/docs
- Streamlit: http://127.0.0.1:8501
- MLflow: http://127.0.0.1:5000 (experiment `environment-smoke-test`)

The API currently exposes only `/health`, with `models_ready: false`.
The UI checks API connectivity. Training and prediction are future work.

## Airflow (Linux container)

[Airflow requires Linux/WSL on Windows](https://airflow.apache.org/docs/apache-airflow/stable/start.html).
Keep it out of the Windows virtual environment. Start a working Docker Desktop
Linux engine, then:

```powershell
docker compose config --quiet
docker compose up -d airflow
docker compose logs -f airflow
docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Open http://127.0.0.1:8080 and use the generated credentials. `standalone` is
for local development. Stop with `docker compose down`; named volume data remains.
The DAG folder is prepared; the training DAG and its Linux ML dependencies will
be added when the training scripts exist. The Windows `.venv` cannot run in Linux.

## Layout

```text
api/                FastAPI application
app/                Streamlit interface
src/brainscan/      Reusable data, training, and inference code
configs/            Model/data configuration
scripts/            Setup and verification commands
data/raw/           Local datasets (ignored by Git)
data/processed/     Prepared datasets (ignored by Git)
models/             Trained artifacts (ignored by Git)
notebooks/          Exploration
reports/            Evaluation reports
airflow/dags/       Orchestration
docs/               Brief and scope decisions
```

## Scope decisions and outstanding prerequisites

- Follow the detailed brief: brain MRI classification with TensorFlow/Keras.
  The opening sentence about blood cells/PyTorch conflicts with those details.
- GoogLeNet/InceptionV1 remains the requested architecture. The pretrained Keras
  implementation and weights must be verified before classification training.
  [Keras Applications](https://keras.io/api/applications/) lists InceptionV3,
  which is a different architecture; no substitution has been made.
- Detection source: [MRI bounding-box dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes).
  Dataset download/access and annotations still need validation.
- Classification dataset source and class names must be confirmed. The previous
  `../BrainScan/Data` remains available as a possible source, without copying it.
- This environment supports CPU execution. Modern native Windows TensorFlow has
  [no CUDA GPU support](https://www.tensorflow.org/install/pip#windows-native);
  GPU training would require a suitable separate environment such as WSL2.
- No dataset, pretrained weight, or training run is created by environment setup.
- Local datasets, environment variables, models, and MLflow storage are excluded
  from Git. Set service URLs in `.env`; no credentials are needed for local setup.
