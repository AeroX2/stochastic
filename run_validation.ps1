# Run experiments validation (all four variants). Uses nanochat's Python 3.12 venv.
# From repo root: .\run_validation.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Nanochat = Join-Path $Root "nanochat"
$VenvPython = Join-Path $Nanochat ".venv312\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Run setup first: .\setup_training_env.ps1"
    exit 1
}

$env:PYTHONPATH = "$Root;$Nanochat"
& $VenvPython -m experiments.run_validation @args
