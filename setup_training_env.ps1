# One-time setup: Python 3.12 venv in nanochat with torch + deps. Run from repo root.
# Requires: pyenv with 3.12.10 (pyenv install 3.12.10)

$ErrorActionPreference = "Stop"
$Nanochat = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "nanochat"

Push-Location $Nanochat
try {
    # Use Python 3.12
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    & python --version
    if ($LASTEXITCODE -ne 0) { throw "python not found" }

    # Create venv if missing
    $venv = Join-Path $Nanochat ".venv312"
    if (-not (Test-Path $venv)) {
        Write-Host "Creating .venv312 with Python 3.12..."
        & python -m venv $venv
    }

    $pip = Join-Path $venv "Scripts\pip.exe"
    $python = Join-Path $venv "Scripts\python.exe"

    # Torch CPU (works without CUDA toolkit). For GPU later: pip install torch --index-url https://download.pytorch.org/whl/cu126
    Write-Host "Installing PyTorch (CPU)..."
    & $pip install torch --index-url "https://download.pytorch.org/whl/cpu" -q

    Write-Host "Installing nanochat dependencies..."
    & $pip install "datasets>=4.0.0" "transformers>=4.57.3" "tiktoken>=0.11.0" "tokenizers>=0.22.0" "regex>=2025.9.1" "scipy>=1.15.3" "wandb>=0.21.3" "zstandard>=0.25.0" "accelerate" "psutil" "python-dotenv" "tabulate" -q

    Write-Host "Done. Run validation: .\run_validation.ps1"
} finally {
    Pop-Location
}
