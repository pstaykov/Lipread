# One-time setup on a Windows machine without GPU (run from the repo folder):
#   powershell -ExecutionPolicy Bypass -File dataset\vm\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

# Visual C++ runtime (needed by torch and mediapipe; missing on fresh Windows Server)
if (-not (Test-Path "$env:WINDIR\System32\msvcp140.dll")) {
    Write-Host "installing Visual C++ runtime"
    Invoke-WebRequest https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile "$env:TEMP\vc_redist.x64.exe"
    Start-Process "$env:TEMP\vc_redist.x64.exe" -ArgumentList "/install /quiet /norestart" -Wait
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# same versions as the main PC (mediapipe 0.10.21 still has mp.solutions; it needs numpy < 2), CPU torch
Write-Host "creating .venv-mine (Python 3.11)"
uv venv -q --allow-existing -p 3.11 .venv-mine
if ($LASTEXITCODE) { throw "uv venv failed" }
$py = ".venv-mine\Scripts\python.exe"
uv pip install -q -p $py numpy==1.26.4 mediapipe==0.10.21 imageio-ffmpeg==0.6.0 protobuf==4.25.9
if ($LASTEXITCODE) { throw "pip install failed" }
uv pip install -q -p $py torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE) { throw "torch install failed" }

# catalog part + vocabulary for this machine (bundestag_data\ is gitignored)
New-Item -ItemType Directory -Force bundestag_data\words, bundestag_data\clips | Out-Null
& $py -c "import gzip, shutil; shutil.copyfileobj(gzip.open('dataset/vm/catalog_1of2.jsonl.gz'), open('bundestag_data/catalog.jsonl', 'wb'))"
if ($LASTEXITCODE) { throw "catalog unpack failed" }
Copy-Item dataset\vm\vocab.csv bundestag_data\words\vocab.csv

& $py dataset\vm\check_env.py
if ($LASTEXITCODE) { throw "environment check failed" }
Write-Host "setup ok -> start with: powershell -ExecutionPolicy Bypass -File dataset\vm\run.ps1"
