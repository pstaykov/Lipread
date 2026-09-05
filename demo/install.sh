#!/usr/bin/env bash
# Sets up a venv, installs deps, launches the server, opens the browser.
# Usage: ./install.sh [port]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT="${1:-8000}"
URL="http://127.0.0.1:${PORT}"
VENV_DIR="venv"

log() { printf '%s\n' "$*"; }

# prefer a python mediapipe has wheels for (3.9-3.12)
version_of() {
  "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null
}

is_mediapipe_compatible() {  # $1 = "X.Y"
  case "$1" in
    3.9|3.10|3.11|3.12) return 0 ;;
    *) return 1 ;;
  esac
}

is_min_supported() {  # $1 = "X.Y", need >=3.9
  local maj="${1%%.*}" min="${1#*.}"
  [ "$maj" -eq 3 ] && [ "$min" -ge 9 ]
}

PYTHON=""
PYTHON_VER=""
PYTHON_COMPAT=0
for cand in python3.12 python3.11 python3.10 python3.9 python3 python; do
  command -v "$cand" >/dev/null 2>&1 || continue
  v="$(version_of "$cand")"
  [ -z "$v" ] && continue
  if is_mediapipe_compatible "$v"; then
    PYTHON="$cand"; PYTHON_VER="$v"; PYTHON_COMPAT=1
    break
  fi
  if [ -z "$PYTHON" ] && is_min_supported "$v"; then
    PYTHON="$cand"; PYTHON_VER="$v"
  fi
done
if [ -z "$PYTHON" ]; then
  log "Kein Python 3.9+ gefunden. Bitte Python installieren: https://www.python.org/downloads/"
  exit 1
fi
log "== GNet Lippenlesen-Demo =="
log "Nutze Python: $PYTHON ($PYTHON_VER)"
if [ "$PYTHON_COMPAT" -ne 1 ]; then
  log "Hinweis: Python $PYTHON_VER wird von 'mediapipe' (Live-Aufnahme) nicht unterstützt (nur 3.9-3.12)."
  log "Für die Live-Aufnahme-Funktion: Python 3.9, 3.10, 3.11 oder 3.12 installieren:"
  log "  https://www.python.org/downloads/  -- die Demo läuft trotzdem, nur ohne 'Selbst aufnehmen'."
fi

# rebuild the venv if it's on a python mediapipe doesn't support
if [ -d "$VENV_DIR" ]; then
  if [ -f "$VENV_DIR/bin/python" ]; then existing_py="$VENV_DIR/bin/python"
  elif [ -f "$VENV_DIR/Scripts/python.exe" ]; then existing_py="$VENV_DIR/Scripts/python.exe"
  else existing_py=""
  fi
  if [ -n "$existing_py" ] && [ "$PYTHON_COMPAT" -eq 1 ]; then
    existing_ver="$(version_of "$existing_py")"
    if ! is_mediapipe_compatible "$existing_ver"; then
      log "Vorhandene venv nutzt Python $existing_ver (kein mediapipe-Wheel) -- wird mit Python $PYTHON_VER neu erstellt..."
      rm -rf "$VENV_DIR"
    fi
  fi
fi
if [ ! -d "$VENV_DIR" ]; then
  log "Erstelle virtuelle Umgebung (./$VENV_DIR) mit Python $PYTHON_VER..."
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

# record deps are best-effort; fall back to serve.py if they don't install
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
elif [ "$PYTHON_COMPAT" -eq 1 ]; then
  log "Hinweis: Live-Aufnahme-Extras konnten nicht installiert werden (siehe Ausgabe oben)."
  log "Die Demo läuft trotzdem -- nur 'Selbst aufnehmen' bleibt ausgeblendet."
else
  log "Die Demo läuft trotzdem -- nur 'Selbst aufnehmen' bleibt ausgeblendet (siehe Hinweis oben)."
fi

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
