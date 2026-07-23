# BWKI Einreichung — German LipNet

Team: lipreaders (pstaykov1, robnok, doribenba69, doribenba1)
Projektname: German LipNet - deutsche Multimodalität für Lippenlesen

---

## Praktische Relevanz und Innovationskraft

### Was ist das Ziel des Projekts? (max. 400)

Ziel ist ein System, das gesprochene deutsche Wörter allein aus der Lippenbewegung erkennt und optional den Ton dazunimmt. Für Deutsch gibt es dafür kaum Modelle, während für Englisch seit Jahren geforscht wird. Wir wollten zeigen, dass sich auf einer einzelnen Consumer-Grafikkarte ein Lippenlesemodell trainieren lässt, das veröffentlichte Ergebnisse auf dem GLips-Datensatz übertrifft.

### Wer kann eure Ergebnisse verwenden? Was ist der Anwendungsfall? (max. 260)

Menschen mit Hörbeeinträchtigung, automatische Untertitelung von Videos und Spracherkennung in lauten Umgebungen oder bei ausgefallenem Mikrofon. Dazu die Forschung: deutschsprachiges Lippenlesen ist gegenüber Englisch stark unterrepräsentiert.

### Was habt ihr entwickelt? (max. 400)

GLipsNet, ein visuelles Wortklassifikationsmodell aus 3D-CNN, ResNet-18, MS-TCN und Transformer-Encoder. Dazu eine komplette Vorverarbeitung mit automatischem Mund-ROI-Zuschnitt, ein Transfer-Verfahren von 500 auf 15 Klassen, einen Beam-Search-Reranker mit Trigramm-Sprachmodell und eine multimodale Erweiterung, die Whisper-Audiofeatures per Cross-Attention einbindet.

### Welchen Umfang hat der Eigenanteil und welche Unterstützung habt ihr erfahren? (max. 400)

Modellarchitektur, Trainingscode, Vorverarbeitung, Ablationen und Auswertung haben wir selbst geschrieben. Übernommen haben wir den GLips-Datensatz, ImageNet-Gewichte für ResNet-18 und den vortrainierten Whisper-Encoder. Als Hilfsmittel haben wir Claude Code für Code-Review und Debugging genutzt; alle Architekturentscheidungen und Experimente stammen von uns.

---

## Eingesetzte Methoden

### Beschreibung eures Datensatzes (max. 600)

GLips (German Lipreading), Schwiebert et al. 2022, Universität Hamburg. Quelle sind öffentliche Aufzeichnungen des Hessischen Landtags. Der Datensatz enthält 249.198 Videoclips von je etwa 1,2 Sekunden zu 500 häufigen deutschen Wörtern, gesprochen von rund 100 verschiedenen Personen, mit fester Aufteilung in Trainings-, Validierungs- und Testdaten. Jeder Clip zeigt das ganze Gesicht in 128x128 Pixeln bei 25 Bildern pro Sekunde und hat eine separate Audiospur. Zusätzlich arbeiten wir mit einer Teilmenge von 15 Wörtern, um direkt mit veröffentlichten Ergebnissen vergleichen zu können.

### Aufbereitung der Daten und sonstige Vorbereitungen (max. 600)

Der wichtigste Schritt war der Mund-ROI-Zuschnitt: die Originalclips zeigen das ganze Gesicht, dadurch lag das Modell anfangs kaum über Zufallsniveau. Mit Mediapipe-Landmarks haben wir für alle 500 Klassen die Mundregion erkannt, zeitlich geglättet und neu zugeschnitten. Jeder Clip wird auf 25 Frames vereinheitlicht, auf 96x96 skaliert und im Training zufällig auf 88x88 beschnitten. Augmentierung: horizontales Spiegeln, Time-Masking von bis zu drei Frames, ImageNet-Normalisierung. Für den Audiozweig haben wir alle Tonspuren einmalig in einen 16 GB großen fp16-Cache dekodiert.

### Beschreibung eurer Methoden (max. 600)

GLipsNet verarbeitet ein Video in vier Stufen: ein 3D-CNN mit Kernel 5x7x7 für kurzfristige Bewegung, ein ResNet-18 pro Einzelbild für visuelle Merkmale, zwei MS-TCN-Blöcke mit parallelen Kerneln 3, 5 und 7 für lokale Zeitmuster und ein Transformer-Encoder mit vier Lagen und Attention-Pooling für den Gesamtkontext. Trainiert wird mit AdamW, Warmup plus Cosine-Schedule, getrennten Lernraten für Backbone und Kopf, Label Smoothing und bfloat16. Als Vergleichsbasis dient ein reines MS-TCN-Modell. Für die 15-Wort-Aufgabe übertragen wir das auf 500 Klassen vortrainierte Frontend.

---

## Euer Ergebnis

### Wie habt ihr ausgewertet? Genauigkeit auf Trainings- und Testdaten? (max. 1000)

Ausgewertet wird auf dem festen Test-Split des Datensatzes, mit identischer Vorverarbeitung für alle Varianten, berichtet werden Top-1, Top-5, Macro-F1 und Konfusionsmatrix. Auf der 15-Wort-Aufgabe erreicht GLipsNet von Grund auf 58,8 Prozent Top-1 auf den Validierungsdaten, mit Transfer vom 500-Klassen-Modell 69,3 Prozent Top-1 und 88,8 Prozent Top-5, auf den Testdaten 66,3 Prozent Top-1. Damit liegen wir über den veröffentlichten Vergleichswerten: NASNetMobile-Transfer 48,4 Prozent, die GLips-Autoren 50,9 Prozent von Grund auf und 54,8 Prozent mit englischem Vortraining. Auf allen 500 Klassen erreichen wir 33,3 Prozent Top-1 und 53,3 Prozent Top-5 bei 0,2 Prozent Zufallsniveau. Die Trainingsgenauigkeit lag durch Augmentierung und Label Smoothing sogar leicht unter der Validierungsgenauigkeit, 55,2 gegenüber 69,3 Prozent, Overfitting war also kein Problem. Ablationen: ohne Mund-ROI fällt Top-1 von 58,8 auf 30,0 Prozent, ohne 3D-CNN-Stem auf 51,7 Prozent.

### Gibt es besondere Anforderungen, um euer Projekt zu nutzen? (max. 400)

Trainiert wurde auf einer einzelnen NVIDIA-GPU mit 12 GB Speicher, ein vollständiger Durchlauf über 500 Klassen dauert mehrere Tage. Der Datensatz belegt mehrere zehn Gigabyte, der Audiocache weitere 16 GB. Für die reine Anwendung reichen eine normale Webcam und eine GPU oder ein aktueller Prozessor. Benötigt werden Python, PyTorch, Mediapipe und Whisper.

---

## Kritische Reflexion

### Auf welche Probleme seid ihr gestoßen? (max. 400)

Anfangs blieb das Modell nahe am Zufall, weil die Clips das ganze Gesicht und nicht nur den Mund zeigen; erst der eigene ROI-Zuschnitt löste das. Danach deuteten auffällig gute Werte auf eine mögliche Überschneidung zwischen den Datensplits hin, weshalb wir alles auf den festen Original-Split umgestellt und komplett neu trainiert haben. Dazu kam knapper GPU-Speicher.

### Was ist das größte Potential eures Projekts? (max. 260)

Ein deutsches Lippenlesemodell, das Bild und Ton kombiniert und auch dann noch versteht, wenn der Ton unbrauchbar ist. Das Grundgerüst lässt sich mit mehr Daten von einzelnen Wörtern auf ganze Sätze erweitern.

### Was ist die größte Schwachstelle eures Projekts? (max. 260)

Das Modell kennt nur 500 Wörter und stammt aus Landtagsaufnahmen: frontale Gesichter, gutes Licht, erwachsene Sprecher. Homophene Wörter, die auf den Lippen gleich aussehen, bleiben grundsätzlich schwer zu unterscheiden.

### Wie würdet ihr das Projekt mit unendlich vielen Ressourcen vorantreiben? (max. 400)

Wir würden einen deutlich größeren deutschen Datensatz mit mehr Sprechern und Alltagsaufnahmen sammeln, statt einzelner Wörter ganze Sätze mit CTC oder einem Sequenz-Decoder vorhersagen, ein großes selbstüberwachtes Audio-Video-Modell nach Art von AV-HuBERT auf Deutsch vortrainieren und das Ergebnis als echtzeitfähige Untertitelung auf dem Handy bereitstellen.

---

## Quellen und Bestätigung der Eigenleistung

Schwiebert et al., A Multimodal German Dataset for Automatic Lip Reading Systems and Transfer Learning, 2022, arXiv:2202.13403 (GLips-Datensatz und Vergleichswerte). Ameer et al., Deep Transfer Learning for Lip Reading Based on NASNetMobile (Vergleichswerte auf der 15-Wort-Aufgabe). Martinez et al., Lipreading using Temporal Convolutional Networks, 2020 (MS-TCN). He et al., Deep Residual Learning for Image Recognition, 2016 (ResNet-18). Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision, 2022 (Whisper). Shi et al., Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction, 2022 (AV-HuBERT). Dokumentation von PyTorch, torchvision und Google Mediapipe. Als generative KI haben wir Claude (Anthropic) über Claude Code für Code-Review, Debugging und Dokumentation eingesetzt.

Wir bestätigen, dass die Modellarchitektur, der Trainings- und Vorverarbeitungscode, die Experimente und die Auswertung dieses Projekts von uns selbst entwickelt wurden. Fremde Beiträge sind der GLips-Datensatz sowie die oben genannten vortrainierten Gewichte und Bibliotheken, die wir als solche gekennzeichnet haben.
