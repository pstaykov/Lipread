# Demo starten

Für die Live-Aufnahme (eigenes Video vor der Kamera):
```
cd demo
python server.py
```
Braucht zusätzlich `flask` und `mediapipe` (siehe `../requirements.txt`).

Ohne Kamera-Feature reicht der einfache statische Server:
```
cd demo
python serve.py
```

Dann im Browser öffnen: **http://localhost:8000**

(Der Ordner muss über HTTP geöffnet werden, nicht als `file://`-Pfad — die Seite
lädt `predictions.json` per `fetch`. Beide Server senden `Cache-Control: no-store`,
damit Änderungen an den Dateien nach einem Neuladen sofort sichtbar sind.)

## Bedienung

- **Selbst aufnehmen**: nur mit `server.py`. Zeigt ein Zielwort; „Aufnahme starten“
  nimmt ~1,8s Video+Ton auf, schneidet die Mundregion per MediaPipe FaceMesh zu
  (derselbe Schritt wie beim Trainingsdatensatz, `lipreading/preprocess_mouth_roi.py`)
  und zeigt danach den zugeschnittenen Clip plus alle drei Modellvorhersagen. Wird
  das Gesicht in weniger als der Hälfte der Frames erkannt, erscheint eine Warnung.
- Heatmap: pro Clip und Modell die Konfidenz für das **tatsächliche** Wort (auch
  wenn nicht Platz 1) — rot (niedrig) über gelb bis grün (hoch).
- Jede Karte darunter zeigt einen echten Testclip mit Originalton und die
  Vorhersage aller drei Modelle: erkanntes Wort, Konfidenz, ob das wahre Wort in
  den Top-5 war, vollständige Top-5-Liste.
- **„Audiobedingung“** schaltet Audio/Multimodal zwischen sauber und Rauschen
  (weiß/Stimmengewirr, 10/5 dB SNR) um — der abgespielte Ton ändert sich mit, man
  hört also genau das Rauschen, das die Modelle bekommen haben (auch bei der
  eigenen Aufnahme). Video bleibt gleich, da es kein Audio nutzt.
- Unten: Genauigkeit über die 16 gezeigten Clips für die aktuelle Bedingung.

## Neu generieren

`build_demo_data.py` erzeugt `predictions.json` und die Clips in `clips/` neu
(führt echte Inferenz mit den Checkpoints aus `../models/` aus). Braucht Zugriff
auf den vollen GLips-Datensatz und läuft daher nur im Hauptrepo, nicht aus diesem
Ordner heraus:

```
python handoff/demo/build_demo_data.py
```

`live_infer.py` (von `server.py` genutzt) braucht dagegen **keinen** GLips-Zugriff
— nur die Checkpoints in `../models/`, die bereits in diesem Ordner liegen.
