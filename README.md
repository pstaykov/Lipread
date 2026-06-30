# Lipread Deutsch - GLips Architektur

Dieses Projekt implementiert eine Lipreading-Pipeline basierend auf dem GLips-Datensatz. Das Modell nutzt eine Kombination aus räumlich-zeitlicher Merkmalsextraktion und Transformer-basierten Sequenz-Modellierung.

## Pflichinformationen zum Datensatz
1. **Title of dataset:** GLips
2. **Authors & paper:** [Visual Speech Recognition for German (2022)](http://arxiv.org/abs/2202.13403)
3. **Source of dataset:** Hessian Parliament (https://hessischer-landtag.de)
4. https://www.fdr.uni-hamburg.de/record/10048

---

## Modellarchitektur

### 1. Input & Augmentierung
Die Eingabe besteht aus RGB-Videoframes, die auf die Lippenregion zugeschnitten sind.
* **Format:** $(B, C, T, H, W)$
    * $B$: Batch-Größe
    * $C$: 3 (RGB-Kanäle)
    * $T$: 25 Frames (gleichmäßig gesampelt oder gepaddet)
    * $H, W$: 96 × 96 → zufälliger Crop auf 88 × 88 (Training), Center-Crop (Validierung)
* **Augmentierung:** Horizontales Flipping (p=0.5), Time Masking (bis zu 3 aufeinanderfolgende Frames auf 0 gesetzt), ImageNet-Normalisierung

### 2. Spatio-Temporal Front-End (3D CNN + ResNet-18)
* **3D CNN:** Kernel $5 \times 7 \times 7$, Stride $(1, 2, 2)$ → BatchNorm → ReLU → MaxPool $(1 \times 3 \times 3)$; extrahiert kurzzeitige Bewegungsmerkmale über die Frame-Abfolge
* **ResNet-18 Backbone:** Stages 3–5 (ab `layer1`) verarbeiten jeden Frame einzeln und extrahieren visuelle Merkmale der Dimension 512
* **Adaptive Average Pooling** kollabiert die räumlichen Dimensionen auf $1 \times 1$
* **Output:** Sequenz von Frame-Features $\in \mathbb{R}^{B \times T \times 512}$

### 3. Projektion
Eine lineare Schicht mit LayerNorm projiziert die CNN-Features auf die Modell-Dimension:
$$512 \rightarrow D_\text{model} = 256$$

### 4. Multi-Scale Temporal Convolutional Network (MS-TCN)
Zwei aufeinanderfolgende MS-TCN-Blöcke erfassen lokale temporale Muster auf verschiedenen Skalen. Jeder Block besteht aus drei parallelen Depthwise-Separable-Konvolutionen mit Kernelgrößen $k \in \{3, 5, 7\}$, deren Ausgaben gemittelt werden:
$$\text{MS-TCN}(x) = \text{LayerNorm}\!\left(x + \frac{1}{3}\sum_{k \in \{3,5,7\}} \text{DW-Conv}_k(x)\right)$$
* **Output:** $\in \mathbb{R}^{B \times T \times 256}$

### 5. Transformer Encoder
Einem erlernbaren Positions-Embedding folgen 4 Transformer-Encoder-Schichten (Pre-Norm):
* **Heads:** 8, **FFN-Dim:** 1024, **Dropout:** 0.1, **Aktivierung:** GELU
* Mean Pooling über die Zeitachse aggregiert die Sequenz zu einem fixen Vektor $\in \mathbb{R}^{B \times 256}$

### 6. Classification Head
Eine lineare Schicht projiziert auf die Zielklassen:
* **Output:** $\in \mathbb{R}^{B \times 500}$ (Logits für 500 deutsche Wörter)

---

## Pipeline Visualisierung
```
Video (B,3,T,96,96)
  → 3D-CNN + ResNet-18          (B, T, 512)
  → Linear + LayerNorm          (B, T, 256)
  → MS-TCN × 2                  (B, T, 256)
  → Positions-Embedding
  → Transformer Encoder × 4    (B, T, 256)
  → Mean Pooling                (B, 256)
  → Classifier                  (B, 500)
```

## Ziel
Folgende Pipeline:
- [ ] Streaming von Video und Audio
- [ ] Zuschnitt auf Mundbereich
- [ ] Bestimmung von Wörtern mit Video und Audio
- [ ] Klassifizierung mit Video und Audio Modell
- [ ] orchestration
- [ ] Captions