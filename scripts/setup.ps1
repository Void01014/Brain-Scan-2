param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Path .tmp -Force | Out-Null
$env:TEMP = Join-Path $PWD '.tmp'
$env:TMP = $env:TEMP
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$venvPython = Join-Path $PWD '.venv/Scripts/python.exe'
& $venvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
$requirements = if (Test-Path requirements-lock.txt) { 'requirements-lock.txt' } else { 'requirements.txt' }
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $venvPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project package installation failed.' }
if (-not (Test-Path '.env')) { Copy-Item .env.example .env }
& $venvPython scripts/check_env.py
if ($LASTEXITCODE -ne 0) { throw 'Environment checks failed.' }
