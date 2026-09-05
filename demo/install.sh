#!/usr/bin/env bash
# One-command setup + launch for the GNet lip-reading demo.
#
#   ./install.sh          # sets up a local venv, installs deps, opens the demo
#   ./install.sh 8080     # use a different port (default 8000)
#
# Safe to re-run: it reuses the existing venv/already-installed packages and
# just (re)starts the server and re-opens the browser.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT="${1:-8000}"
URL="http://127.0.0.1:${PORT}"
VENV_DIR="venv"

log() { printf '%s\n' "$*"; }

# ---- 1. find a python 3 interpreter -----------------------------------
PYTHON=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 \
     && "$cand" -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' >/dev/null 2>&1; then
    PYTHON="$cand"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  log "Kein Python 3.9+ gefunden. Bitte Python installieren: https://www.python.org/downloads/"
  exit 1
fi
log "== GNet Lippenlesen-Demo =="
log "Nutze Python: $("$PYTHON" --version 2>&1)"

# ---- 2. virtualenv, isolated from any other Python setup on this machine ----
if [ ! -d "$VENV_DIR" ]; then
  log "Erstelle virtuelle Umgebung (./$VENV_DIR)..."
  "$PYTHON" -m venv "$VENV_DIR"
fi
if [ -f "$VENV_DIR/bin/python" ]; then
  VENV_PY="$VENV_DIR/bin/python"
elif [ -f "$VENV_DIR/Scripts/python.exe" ]; then
  VENV_PY="$VENV_DIR/Scripts/python.exe"
else
  log "Virtuelle Umgebung konnte nicht erstellt werden."
  exit 1
fi
"$VENV_PY" -m pip install --upgrade pip -q

# ---- 3. install deps. Core deps (video+audio comparison demo) must succeed;
# the live-webcam-recording extras (flask/mediapipe/opencv) are best-effort --
# mediapipe's pinned build isn't available for every Python version/platform,
# so if that install fails we fall back to the static comparison demo instead
# of breaking the whole thing. ----
CORE_DEPS=(torch torchvision "numpy<2" tqdm openai-whisper imageio_ffmpeg imageio)
RECORD_DEPS=(flask "mediapipe==0.10.21" opencv-python)

log "Installiere Kernabhängigkeiten (erster Lauf kann mehrere Minuten dauern)..."
if ! "$VENV_PY" -m pip install -q "${CORE_DEPS[@]}"; then
  log "FEHLER: Kernabhängigkeiten (PyTorch/Whisper/...) konnten nicht installiert werden."
  log "Bitte die Ausgabe oben prüfen (z.B. Python-Version zu neu/alt für PyTorch)."
  exit 1
fi

SERVER_SCRIPT="serve.py"
log "Installiere Live-Aufnahme-Extras (flask, mediapipe, opencv)..."
if "$VENV_PY" -m pip install -q "${RECORD_DEPS[@]}"; then
  SERVER_SCRIPT="server.py"
else
  log "Hinweis: Live-Aufnahme-Extras konnten auf diesem Rechner nicht installiert werden"
  log "(mediapipe hat oft keine Wheels für die neueste Python-Version)."
  log "Die Demo läuft trotzdem -- nur 'Selbst aufnehmen' bleibt ausgeblendet."
fi

# ---- 4. (re)start the server -----------------------------------------
is_up() {
  "$VENV_PY" - "$PORT" <<'PYEOF' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}", timeout=1)
PYEOF
}

if [ -f ".server.pid" ]; then
  OLD_PID="$(cat .server.pid 2>/dev/null || true)"
  if [ -n "$OLD_PID" ]; then
    kill "$OLD_PID" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f .server.pid
fi

log "Starte $SERVER_SCRIPT auf Port $PORT..."
"$VENV_PY" "$SERVER_SCRIPT" "$PORT" > server.log 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > .server.pid
trap 'kill "$SERVER_PID" >/dev/null 2>&1; rm -f .server.pid' EXIT INT TERM

log "Warte auf den Server (lädt Modelle, kann bis zu ~1 Minute dauern)..."
UP=0
for _ in $(seq 1 60); do
  if is_up; then UP=1; break; fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then break; fi
  sleep 1
done

if [ "$UP" -ne 1 ]; then
  log "Server ist nicht rechtzeitig gestartet. Details in server.log:"
  tail -n 30 server.log 2>/dev/null || true
  exit 1
fi

log "Demo läuft: $URL"

# ---- 5. open the browser -----------------------------------------------
if command -v open >/dev/null 2>&1; then
  open "$URL" >/dev/null 2>&1
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1
elif command -v cmd.exe >/dev/null 2>&1; then
  cmd.exe /c start "" "$URL" >/dev/null 2>&1
elif command -v start >/dev/null 2>&1; then
  start "$URL" >/dev/null 2>&1
else
  log "Bitte manuell im Browser öffnen: $URL"
fi

log ""
log "Fertig. Strg+C in diesem Fenster beendet den Server wieder."
wait "$SERVER_PID" 2>/dev/null
