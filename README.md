# Lipread Deutsch - GLips Architektur

Dieses Projekt implementiert eine Lipreading-Pipeline basierend auf dem GLips-Datensatz. Das Modell nutzt eine Kombination aus räumlich-zeitlicher Merkmalsextraktion und Transformer-basierten Sequenz-Modellierung.

## Pflichinformationen zum Datensatz
1. **Title of dataset:** GLips
2. **Authors & paper:** [Visual Speech Recognition for German (2022)](http://arxiv.org/abs/2202.13403)
3. **Source of dataset:** Hessian Parliament (https://hessischer-landtag.de)
4. https://www.fdr.uni-hamburg.de/record/10048

---

## Modellarchitektur

### 1. Input
Die Eingabe besteht aus Graustufen- oder RGB-Videoframes, die auf die Lippenregion zugeschnitten sind.
* **Format:** $(B, C, T, H, W)$
    * $B$: Batch-Größe
    * $C$: 3 (RGB-Kanäle)
    * $T$: Anzahl der Frames (Zeitachse)
    * $H, W$: 96 x 96 (Höhe und Breite)

### 2. Spatio-Temporal Front-End (3D CNN + ResNet)
Da Lippenbewegungen sowohl eine zeitliche als auch eine präzise räumliche Komponente haben, besteht das Front-End aus zwei Teilen:
* **3D CNN:** Ein initialer 3D-Kernel (z. B. $5 \times 7 \times 7$) extrahiert kurzzeitige Bewegungsmerkmale direkt aus der Frame-Abfolge.
* **ResNet-18:** Ein 2D-ResNet fungiert als Feature-Extractor. Es verarbeitet die Zeitdimension Frame für Frame (oder über die 3D-Features), um hochdimensionale visuelle Merkmale zu extrahieren.
* **Output:** $V = (D, T')$, wobei $D$ (z. B. 512) die Merkmalsdimension pro Frame ist.

### 3. Transformer Encoder
Transformer modellieren die Zusammenhänge über das gesamte Wort hinweg. Durch Self-Attention lernt das Modell, welche Phasen der Lippenbewegung für die Unterscheidung der 500 Wörter entscheidend sind.
* **Output:** $V = (D, T')$ (Kontextualisierte Repräsentationen).

### 4. Mean Pooling
Um die Informationen der gesamten Sequenz zu aggregieren, wird der Mittelwert über die Zeitachse $T'$ berechnet:
$$\frac{1}{T'} \sum_{t=1}^{T'} V_t = \bar{V}$$
* **Output:** Ein fixer Merkmalsvektor $V = (D)$.

### 5. Classification Head (Neural Network)
Ein abschließendes Fully-Connected-Netzwerk projiziert den Vektor auf die Anzahl der Zielklassen.
* **Output:** $V = (500)$ (Logits für die Wort-Klassifizierung).

---

## Pipeline Visualisierung
Die Architektur folgt dem Prinzip: **Lokale Bewegung (3D-CNN) $\rightarrow$ Visuelle Details (ResNet) $\rightarrow$ Globaler Kontext (Transformer).**

## Ziel
Folgende Pipeline:
- [ ] Streaming von Video und Audio
- [ ] Zuschnitt auf Mundbereich
- [ ] Bestimmung von Wörtern mit Video und Audio
- [ ] Klassifizierung mit Video und Audio Modell
- [ ] orchestration
- [ ] Captions