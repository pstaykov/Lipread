# German LipNet — deutsche Multimodalität für Lippenlesen

Erkennung gesprochener deutscher Wörter aus der Lippenbewegung, optional
kombiniert mit dem Ton. Trainiert auf dem GLips-Datensatz (500 Wörter) auf einer
einzelnen Consumer-Grafikkarte.

**Team lipreaders** — pstaykov1, robnok, doribenba1

---

## Demo starten

**Windows:** `install.bat` im Ordner `demo/` doppelklicken
**macOS / Linux:**

```bash
cd demo
chmod +x install.sh
./install.sh
```

Falls macOS die direkte Ausführung von install.sh verhindert, stattdessen:

```bash
bash install.sh
```

Das Skript legt eine isolierte Python-Umgebung an, installiert alles Nötige,
startet den Server und öffnet die Demo im Browser (`http://localhost:8000`). Der
erste Start dauert einige Minuten, weil PyTorch heruntergeladen wird; danach
startet sie in Sekunden. Beenden mit Strg+C.

Voraussetzung ist nur **Python 3.9+** ([python.org](https://www.python.org/downloads/),
unter Windows im Installer "Add python.exe to PATH" ankreuzen). Alles Weitere
landet in einem lokalen `venv/`-Ordner im `demo/`-Verzeichnis und berührt nichts
anderes am System. Eine GPU wird nicht gebraucht, die Modelle laufen auf der CPU.
Modellgewichte und Clips liegen bereits bei — es muss nichts nachgeladen werden.

Manuell geht es auch: `pip install -r demo/requirements.txt`, dann
`python server.py` (oder `python serve.py` ohne Webcam-Funktion). Die Seite muss
über HTTP laufen, nicht als `file://` geöffnet werden.

### Was die Demo zeigt

- **Vergleich der drei Modelle** (nur Video, nur Audio, multimodal) auf 16
  Test-Clips: das echte Wort, die Top-1-Vorhersage jedes Modells mit Konfidenz
  und die Top-5. Die 16 Clips sind eine Zufallsziehung mit festem Seed, nicht
  handverlesen — einer davon (`haushalt`) wird von allen dreien falsch erkannt.
- **Rausch-Auswahl:** ein Dropdown schaltet zwischen sauberem Audio und
  zusätzlichem weißem bzw. Stimmengewirr-Rauschen (10 dB / 5 dB SNR). Hier zeigt
  sich der Nutzen der Fusion am deutlichsten: das reine Audio-Modell bricht ein,
  das multimodale hält sich.
- **Selbst aufnehmen:** ein Wort in die Webcam sprechen. Die Mundregion wird mit
  derselben MediaPipe-Pipeline zugeschnitten, durch die auch die Trainingsdaten
  gelaufen sind, und zurückgezeigt — man sieht also, was die Modelle sehen.
  Lässt sich `mediapipe` auf der Maschine nicht installieren, fällt das Skript
  automatisch auf die statische Demo ohne diesen Teil zurück.

---

## Ergebnisse

Alle Werte auf dem vollen 500-Wort-Vokabular, identische Klassenordnung, also
direkt vergleichbar:

| Modell | Top-1 | Top-5 |
|---|---|---|
| nur Video (GNet) | 34,2 % | 53,9 % |
| nur Audio (Whisper-Probe) | 61,2 % | 76,9 % |
| **multimodal** | **72,0 %** | **79,8 %** |

Nur Video und multimodal sind auf dem zurückgehaltenen **Test**-Split gemessen,
nur Audio auf **Val** (der Val/Test-Abstand lag beim Schwestermodell unter einem
halben Punkt).

Auf der 15-Wort-Teilmenge erreicht GNet **59,3 %** von Grund auf und **69,5 %**
mit Transfer aus dem 500-Klassen-Modell — vor allen bisher veröffentlichten
Ergebnissen auf diesem Korpus. Das multimodale Ergebnis ist das erste
veröffentlichte audiovisuelle Resultat für GLips überhaupt.

---

## Das Modell

**Visueller Zweig (GNet).** Ein 3D-CNN (Kernel 5×7×7) erfasst kurzfristige
Mundbewegung über aufeinanderfolgende Frames, ein ResNet-18 verarbeitet
anschließend jedes Frame einzeln zu visuellen Merkmalen. Zwei MS-TCN-Blöcke mit
parallelen Kerneln (3, 5, 7) modellieren lokale Zeitmuster auf mehreren Skalen,
ein Transformer-Encoder mit vier Lagen den Gesamtkontext. Nach Pooling über die
Zeitachse gibt ein linearer Kopf die Logits über die 500 Wörter aus.

**Audio-Zweig.** Der vortrainierte Whisper-`base`-Encoder liefert die
Audio-Features und bleibt eingefroren und unangepasst.

**Fusion.** Die Audio-Features werden mit einem gemeinsamen Transformer in den visuellen
Token-Strom eingebunden, danach wird end-to-end feinjustiert. Verglichen haben
wir das gegen Late Fusion, Concatenation und Cross-Attention;
 gemeinsamer Transformer gewinnt (`multimodal/results/fusion_comparison.csv`).

**Vorverarbeitung.** Der wichtigste Schritt: die Originalclips zeigen das ganze
Gesicht, damit lag das Modell kaum über Zufallsniveau. Mit MediaPipe-Landmarks
wird die Mundregion erkannt, zeitlich geglättet und zugeschnitten. Jeder Clip
wird auf 25 Frames und 96×96 vereinheitlicht, im Training zufällig auf 88×88
beschnitten. Augmentierung: horizontales Spiegeln, Time-Masking,
ImageNet-Normalisierung.

**Training.** AdamW, Warmup plus Cosine-Schedule, getrennte Lernraten für
Backbone und Kopf, Label Smoothing, bfloat16.

Die mitgelieferten Gewichte speichern ihre großen Tensoren als float16, damit
das Paket unter die Größengrenze passt; BatchNorm-Statistiken bleiben float32
und PyTorch rechnet beim Laden wieder hoch. Top-1 und Top-5 sind dadurch
unverändert.

---

## Was liegt wo

- **`demo/`** — die lauffähige Demo: Modellgewichte, Clips und Code,
  vollständig eigenständig. Hier anfangen.
- **`lipreading/`** — das visuelle Modell: Mund-ROI-Zuschnitt
  (`preprocess_mouth_roi.py`), GNet-Architektur und Trainingsskripte
  (`Transformer_based/`), die MS-TCN-Vergleichsbasis ohne Transformer
  (`mstcn_baseline/`) sowie Auswertung, Ablationen und Plots (`analysis/`).
- **`multimodal/`** — die Fusion von Video und Audio: Cross-Attention-Training
  (`train.py`), die vier verglichenen Fusionsvarianten (`fusion_heads.py`),
  Audio-Vorabdekodierung (`build_audio_cache.py`), Auswertung unter Rauschen
  (`snr_eval.py`), alle Messwerte als CSV (`results/`) und die Abbildungen
  (`plots/`).
- **`camera_tracking/`** — Kamera-Vorschau zum Prüfen des Mund-Zuschnitts.

---

## Datensatz

**GLips** (German Lipreading), Schwiebert et al. 2022, Universität Hamburg:
250.000 Clips von je etwa 1,2 Sekunden zu 500 häufigen deutschen Wörtern,
gesprochen von rund 100 Personen, aus öffentlichen Aufzeichnungen des Hessischen
Landtags, mit fester Aufteilung in Trainings-, Validierungs- und Testdaten.

- Paper: [Visual Speech Recognition for German (2022)](http://arxiv.org/abs/2202.13403)
- Datensatz: https://www.fdr.uni-hamburg.de/record/10048
- Quelle: Hessischer Landtag (https://hessischer-landtag.de)

Der Datensatz ist hier wegen seiner Größe **nicht** enthalten. Die Demo braucht
ihn auch nicht.

## Einschränkungen

Das Vokabular ist auf 500 Wörter begrenzt und stammt ausschließlich aus dem
Parlamentskontext. Die stock-Aufteilung von GLips ist nicht sprecher-disjunkt —
die Werte sind also innerhalb des Korpus zu lesen, nicht als
sprecherunabhängig. Pro Modell gibt es nur einen Seed. Der Audio-Zweig ist
Whisper `base`, überwiegend auf Englisch vortrainiert und eingefroren auf
Deutsch angewandt. Die praktische Genauigkeit bei eigenen Webcam-Aufnahmen liegt
spürbar unter den Testwerten — in der Demo ist das direkt zu erleben.

## Quellen

- Schwiebert et al., *A Multimodal German Dataset for Automatic Lip Reading
  Systems and Transfer Learning*, 2022, arXiv:2202.13403 — GLips-Datensatz und
  Vergleichswerte
- Ameer et al., *Deep Transfer Learning for Lip Reading Based on NASNetMobile* —
  Vergleichswerte auf der 15-Wort-Aufgabe
- Martinez et al., *Lipreading using Temporal Convolutional Networks*, 2020 —
  MS-TCN

Modellarchitektur, Trainingscode, Vorverarbeitung, Ablationen und Auswertung
stammen von uns. Übernommen haben wir den GLips-Datensatz, ImageNet-Gewichte für
ResNet-18 und den vortrainierten Whisper-Encoder. Als Hilfsmittel haben wir
Coding-Agenten (Claude) für Code-Review, Debugging und teilweise Implementation
genutzt; alle Architekturentscheidungen und Experimente stammen von uns.
