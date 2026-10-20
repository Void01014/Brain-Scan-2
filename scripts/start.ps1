param([Parameter(Mandatory=$true)][ValidateSet('api','ui','mlflow')][string]$Service)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$venvPython = Join-Path $PWD '.venv/Scripts/python.exe'
switch ($Service) {
    'api' { & $venvPython -m uvicorn api.main:app --host 127.0.0.1 --port 8000 }
    'ui' { & $venvPython -m streamlit run app/main.py --server.address 127.0.0.1 --browser.gatherUsageStats false }
    'mlflow' { & $venvPython -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000 }
}
if ($LASTEXITCODE -ne 0) { throw "Service exited with code $LASTEXITCODE" }
