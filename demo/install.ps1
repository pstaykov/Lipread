# Windows version of install.sh. Usage: .\install.ps1 [-Port 8080]
param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Write-Info($msg) { Write-Host $msg }

Write-Info "== GNet Lippenlesen-Demo (Windows) =="

# prefer a python mediapipe has wheels for (3.9-3.12); skips the MS Store stub
$MediapipeCompatVersions = @('3.9', '3.10', '3.11', '3.12')

function Get-PyVersion {
    param([string]$Exe, [string[]]$ExtraArgs)
    try {
        $out = & $Exe @ExtraArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        return ($out | Select-Object -Last 1).Trim()
    } catch {
        return $null
    }
}

function Test-MinSupported {
    param([string]$Ver)
    if (-not $Ver) { return $false }
    $parts = $Ver -split '\.'
    return ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 9)
}

$Candidates = @()
foreach ($v in $MediapipeCompatVersions[($MediapipeCompatVersions.Length - 1)..0]) {
    $Candidates += @{ Exe = 'py'; Args = @("-$v") }
}
$Candidates += @{ Exe = 'py'; Args = @('-3') }
$Candidates += @{ Exe = 'python'; Args = @() }
$Candidates += @{ Exe = 'python3'; Args = @() }

$Interpreter = $null
$InterpArgs = @()
$InterpVer = $null
$PythonCompat = $false
foreach ($c in $Candidates) {
    if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
    $v = Get-PyVersion -Exe $c.Exe -ExtraArgs $c.Args
    if (-not $v) { continue }
    if ($MediapipeCompatVersions -contains $v) {
        $Interpreter = $c.Exe; $InterpArgs = $c.Args; $InterpVer = $v; $PythonCompat = $true
        break
    }
    if (-not $Interpreter -and (Test-MinSupported $v)) {
        $Interpreter = $c.Exe; $InterpArgs = $c.Args; $InterpVer = $v
    }
}
if (-not $Interpreter) {
    Write-Info "Kein Python 3.9+ gefunden. Bitte installieren: https://www.python.org/downloads/"
    Write-Info "(Beim Installer unbedingt 'Add python.exe to PATH' anhaken.)"
    exit 1
}
Write-Info "Nutze Python: $Interpreter $InterpArgs ($InterpVer)"
if (-not $PythonCompat) {
    Write-Info "Hinweis: Python $InterpVer wird von 'mediapipe' (Live-Aufnahme) nicht unterstuetzt (nur 3.9-3.12)."
    Write-Info "Fuer die Live-Aufnahme-Funktion: Python 3.9, 3.10, 3.11 oder 3.12 installieren:"
    Write-Info "  https://www.python.org/downloads/  -- die Demo laeuft trotzdem, nur ohne 'Selbst aufnehmen'."
}

# rebuild the venv if it's on a python mediapipe doesn't support
function Remove-VenvDir {
    # retries: Windows sometimes briefly locks a just-used venv's python.exe
    param([string]$Path)
    for ($i = 0; $i -lt 5; $i++) {
        try {
            Remove-Item -Recurse -Force $Path -ErrorAction Stop
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

$VenvDir = Join-Path $PSScriptRoot 'venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
if ((Test-Path $VenvPy) -and $PythonCompat) {
    $existingVer = Get-PyVersion -Exe $VenvPy -ExtraArgs @()
    if (-not ($MediapipeCompatVersions -contains $existingVer)) {
        Write-Info "Vorhandene venv nutzt Python $existingVer (kein mediapipe-Wheel) -- wird mit Python $InterpVer neu erstellt..."
        if (-not (Remove-VenvDir -Path $VenvDir)) {
            Write-Info "Konnte die alte venv nicht entfernen (Datei gesperrt?) -- fahre mit der vorhandenen venv fort."
        }
    }
}
if (-not (Test-Path $VenvPy)) {
    Write-Info "Erstelle virtuelle Umgebung (.\venv) mit Python $InterpVer..."
    & $Interpreter @InterpArgs -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Virtuelle Umgebung konnte nicht erstellt werden."
        exit 1
    }
}
& $VenvPy -m pip install --upgrade pip -q

# record deps are best-effort; fall back to serve.py if they don't install
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
} elseif ($PythonCompat) {
    Write-Info "Hinweis: Live-Aufnahme-Extras konnten nicht installiert werden (siehe Ausgabe oben)."
    Write-Info "Die Demo läuft trotzdem -- nur 'Selbst aufnehmen' bleibt ausgeblendet."
} else {
    Write-Info "Die Demo läuft trotzdem -- nur 'Selbst aufnehmen' bleibt ausgeblendet (siehe Hinweis oben)."
}

# detached so it keeps running after this script/window closes
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

Start-Process $Url

Write-Info ""
Write-Info "Fertig. Der Server läuft im Hintergrund weiter (PID $($proc.Id))."
Write-Info "Zum Beenden:  Stop-Process -Id $($proc.Id)   (oder install.ps1/install.bat erneut ausführen, das ersetzt ihn)"
