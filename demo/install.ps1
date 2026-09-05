# One-command setup + launch for the GNet lip-reading demo, Windows/PowerShell
# version of install.sh.
#
#   .\install.ps1          # sets up a local venv, installs deps, opens the demo
#   .\install.ps1 -Port 8080
#
# Safe to re-run: it reuses the existing venv/already-installed packages and
# just (re)starts the server and re-opens the browser. Most people should just
# double-click install.bat, which calls this script for you.

param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Write-Info($msg) { Write-Host $msg }

Write-Info "== GNet Lippenlesen-Demo (Windows) =="

# ---- 1. find a python 3.9+ interpreter -----------------------------------
# Skips the Microsoft Store "python" stub some machines ship, which exits
# with an error instead of a version when actually invoked.
function Test-PythonCandidate {
    param([string]$Exe, [string[]]$ExtraArgs)
    try {
        $out = & $Exe @ExtraArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $parts = ($out | Select-Object -Last 1).Trim() -split '\.'
        return ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 9)
    } catch {
        return $false
    }
}

$Candidates = @(
    @{ Exe = 'py'; Args = @('-3') },
    @{ Exe = 'python'; Args = @() },
    @{ Exe = 'python3'; Args = @() }
)
$Interpreter = $null
$InterpArgs = @()
foreach ($c in $Candidates) {
    if (Get-Command $c.Exe -ErrorAction SilentlyContinue) {
        if (Test-PythonCandidate -Exe $c.Exe -ExtraArgs $c.Args) {
            $Interpreter = $c.Exe
            $InterpArgs = $c.Args
            break
        }
    }
}
if (-not $Interpreter) {
    Write-Info "Kein Python 3.9+ gefunden. Bitte installieren: https://www.python.org/downloads/"
    Write-Info "(Beim Installer unbedingt 'Add python.exe to PATH' anhaken.)"
    exit 1
}
Write-Info "Nutze Python: $Interpreter $InterpArgs"

# ---- 2. virtualenv, isolated from any other Python setup on this machine ----
$VenvDir = Join-Path $PSScriptRoot 'venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $VenvPy)) {
    Write-Info "Erstelle virtuelle Umgebung (.\venv)..."
    & $Interpreter @InterpArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Virtuelle Umgebung konnte nicht erstellt werden."
        exit 1
    }
}
& $VenvPy -m pip install --upgrade pip -q

# ---- 3. install deps. Core deps (video+audio comparison demo) must succeed;
# the live-webcam-recording extras (flask/mediapipe/opencv) are best-effort --
# mediapipe's pinned build isn't available for every Python version, so if
# that install fails we fall back to the static comparison demo instead of
# breaking the whole thing. ----
$CoreDeps = @('torch', 'torchvision', 'numpy<2', 'tqdm', 'openai-whisper', 'imageio_ffmpeg', 'imageio')
$RecordDeps = @('flask', 'mediapipe==0.10.21', 'opencv-python')

Write-Info "Installiere Kernabhängigkeiten (erster Lauf kann mehrere Minuten dauern)..."
& $VenvPy -m pip install -q @CoreDeps
if ($LASTEXITCODE -ne 0) {
    Write-Info "FEHLER: Kernabhängigkeiten (PyTorch/Whisper/...) konnten nicht installiert werden."
    Write-Info "Bitte die Ausgabe oben prüfen (z.B. Python-Version zu neu/alt für PyTorch)."
    exit 1
}

$ServerScript = 'serve.py'
Write-Info "Installiere Live-Aufnahme-Extras (flask, mediapipe, opencv)..."
& $VenvPy -m pip install -q @RecordDeps
if ($LASTEXITCODE -eq 0) {
    $ServerScript = 'server.py'
} else {
    Write-Info "Hinweis: Live-Aufnahme-Extras konnten auf diesem Rechner nicht installiert werden"
    Write-Info "(mediapipe hat oft keine Wheels für die neueste Python-Version)."
    Write-Info "Die Demo läuft trotzdem -- nur 'Selbst aufnehmen' bleibt ausgeblendet."
}

# ---- 4. (re)start the server, detached so it keeps running after this
# script (and even this window) closes ----
$PidFile = Join-Path $PSScriptRoot '.server.pid'
if (Test-Path $PidFile) {
    $oldPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($oldPid) {
        Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

$OutLog = Join-Path $PSScriptRoot 'server.log'
$ErrLog = Join-Path $PSScriptRoot 'server.err.log'
Write-Info "Starte $ServerScript auf Port $Port..."
$proc = Start-Process -FilePath $VenvPy -ArgumentList @($ServerScript, $Port) `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii -NoNewline

# ---- 5. wait for it to come up -----------------------------------------
$Url = "http://127.0.0.1:$Port"
Write-Info "Warte auf den Server (lädt Modelle, kann bis zu ~1 Minute dauern)..."
$Up = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1
        if ($resp.StatusCode -eq 200) { $Up = $true; break }
    } catch {}
    if ($proc.HasExited) { break }
    Start-Sleep -Seconds 1
}

if (-not $Up) {
    Write-Info "Server ist nicht rechtzeitig gestartet. Details in server.log / server.err.log:"
    Get-Content $ErrLog -Tail 30 -ErrorAction SilentlyContinue
    exit 1
}

Write-Info "Demo läuft: $Url"

# ---- 6. open the browser -----------------------------------------------
Start-Process $Url

Write-Info ""
Write-Info "Fertig. Der Server läuft im Hintergrund weiter (PID $($proc.Id))."
Write-Info "Zum Beenden:  Stop-Process -Id $($proc.Id)   (oder install.ps1/install.bat erneut ausführen, das ersetzt ihn)"
