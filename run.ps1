# Lance l'application localement.
#
#   .\run.ps1
#
# Le serveur est lie explicitement a IPv4 (127.0.0.1). Sur Windows, la double
# pile IPv6 utilisee par defaut peut provoquer une erreur asyncio
# "WinError 64 - Le nom reseau specifie n'est plus disponible" qui fait tomber
# le serveur. Cette liaison n'expose en plus l'application que sur votre machine.
#
# Le fichier .streamlit/config.toml ne fixe volontairement pas l'adresse, afin
# de ne pas gener le deploiement sur Streamlit Cloud.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Environnement virtuel absent : creation en cours..." -ForegroundColor Yellow
    python -m venv .venv
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
}

Write-Host "Application disponible sur http://127.0.0.1:8501" -ForegroundColor Green
& $python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
