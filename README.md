# Lipread Deutsch – Jugend forscht 2027

Lippenlese-Modelle werden meist an Benchmark-Werten auf bereits gesehenen Sprechern gemessen. Dieses Projekt lässt die Rangliste beiseite und stellt drei grundlegendere Fragen:

1. Wie viel von Sprache ist überhaupt auf den Lippen sichtbar?
2. Reichen wenige Lippenkoordinaten statt Videoframes aus?
3. Wie viel verraten Lippendaten darüber, wer spricht?

Um jede Variable zu kontrollieren, wird ein eigener Datensatz aus Plenarvideos des Deutschen Bundestages aufgebaut. Das Ergebnis ist ein Zielkonflikt: wie viel Erkennungsgenauigkeit man für wie viel Privatsphäre aufgibt, und ob ein Modell, das jeden Mund auf einen neutralen Mund abbildet, diesen Zielkonflikt verschieben kann.

## Bausteine
| # | Baustein | Frage |
|---|---|---|
| 1 | **Eigener Datensatz** | Bundestagsreden mit exakten Sprecher-Labels, von Anfang an sprecherdisjunkt |
| 2 | **Sichtbarkeitsgrenze** | Theoretische Obergrenze der Genauigkeit |
| 3 | **Landmarken vs. Pixel vs. Audio** | Welche Eingabe reicht aus? |
| 4 | **Identitäts-Leakage** | Wie gut identifiziert jede Eingabe den Sprecher? |
| 5 | **Neutraler-Mund-Anonymisierer** | Lippen jedes Sprechers auf einen gemeinsamen Mund abgebildet |

### 1. Eigener Datensatz aus Bundestagsvideos
Die Sprecher-Labels von GLips sind heuristisch und nicht rekonstruierbar. Ein eigener Datensatz erlaubt es, Wörter und Sprecher passend zu den Fragen zu wählen.

**Stand:** BTLips ist gebaut: 500 Wörter × 500 Clips, 400/50/50 Clips pro Wort, global sprecherdisjunkt (jeder der 1.372 Sprecher in genau einem von train/val/test), FaceMesh vorberechnet. Details: [`docs/DATASET.md`](docs/DATASET.md).

* **Sprecher-Labels:** Die Mediathek veröffentlicht jede Rede als eigenes Video. Die [Open Parliament TV API](https://github.com/OpenParliamentTV/OpenParliamentTV-Architecture/blob/main/PLATFORM-DATAOBJECTS.md) verknüpft jede Rede mit Sprecher (Wikidata-ID), Sitzung und Transkript. OCR des Namensbanners dient als Gegenprobe.
* **Wort-Zeitstempel:** Forced Alignment (MFA German oder WhisperX) liefert Wort- und Phonemgrenzen. Da Protokolle redigiert und nicht wörtlich sind, wird gegen ASR-Ausgabe ausgerichtet oder nur Wörter behalten, bei denen ASR und Protokoll übereinstimmen.
* **Clips:** Face Tracking → Mund-Zuschnitt, Clips fester Länge um jedes Zielwort, dazu FaceMesh-Landmarken und Audio.
* **Vokabular nach Plan:** bewusst visemgleiche Paare und Beinahe-Paare, damit die Sichtbarkeitsgrenze eine testbare Vorhersage wird.
* **Split:** nach Sprecher, ca. 80/10/10, grob ausgeglichen nach Partei und Geschlecht.
* **Veröffentlichung:** Medien-IDs, Zeitstempel, Crop-Boxen und das Build-Skript, **keine Clips**.
* **Optional:** GLips als externer Test (Training auf Bundestag, Test auf Landtagsvideos).

### 2. Sichtbarkeitsgrenze
Jedes Wort wird auf Phoneme und dann auf Viseme abgebildet. Wörter mit identischer Visemfolge (Homophene) sind nicht unterscheidbar. Bei ausgeglichenen Klassen ist Raten innerhalb einer Gruppe optimal:

$$\text{Acc}_{\max} = \frac{\#\,\text{Homophen-Gruppen}}{\#\,\text{Wörter}}$$

* **Zungen-Parameter $n$:** Zungenkontraste gelten mit Wahrscheinlichkeit $n$ als sichtbar, Lippenkontraste immer, Stimmhaftigkeit und Nasalität nie (p/b/m). Die Grenze wird für $n = 0$ bis $1$ geplottet.
* **FaceMesh-Grenze (eigenes Maß):** Wie gut trennen normalisierte FaceMesh-Lippenpunkte jedes Phonempaar? Nicht trennbare Paare werden zusammengelegt und die Homophene neu berechnet. Das begrenzt jedes reine Landmarken-Modell.
* **Validierung:** Die Verwechslungen der Modelle sollten innerhalb der vorhergesagten Homophen-Gruppen liegen, geprüft an den eingebauten Paaren.
* **Vokabulargröße:** Grenze für wachsende Wortlisten, da lange deutsche Wörter selten kollidieren.

Eine deutsche Visem-Zuordnung gibt es von [Weiss & Aschenberner 2005](https://www.isca-archive.org/interspeech_2005/weiss05_interspeech.html). [Bear & Harvey 2017](https://ueaeprints.uea.ac.uk/id/eprint/64350/) zeigen, dass die Wahl der Zuordnung eine Rolle spielt.

### 3. Landmarken vs. Pixel vs. Audio
Gleiches Back-End für jede Eingabe, sodass sich nur die Eingabe ändert.

* **Eingaben:** Pixel, FaceMesh (roh → Procrustes-ausgerichtet → Sprechermittel entfernt), Audio und deren Fusionen.
* **„Reicht aus“** wird vorab festgelegt, z. B. Landmarken erreichen X % der Pixel-Genauigkeit auf ungesehenen Sprechern.
* **Erwartung:** große Lücke. Auf dem englischen LRW erreichen Landmarken-Modelle ca. 61 % gegenüber ca. 89–92 % für Pixel.
* Die FaceMesh-Grenze aus Baustein 2 zeigt, wie nah das Landmarken-Modell an sein Limit kommt.
* **Modell:** NASNet-Mobile-Encoder pro Frame → DC-TCN-Back-End (Ma et al., LRW). Acht Eingabevarianten (RGB mit Gesichts- und Mund-Zuschnitt, RGB mit FaceMesh-Overlay, gerendertes Mesh Gesicht/Mund, Koordinaten Gesicht/Mund, RGB + Koordinaten). Details und Begründung: [`docs/MODELS.md`](docs/MODELS.md).

### 4. Identitäts-Leakage
Für jede Eingabe wird ein Sprecher-Klassifikator trainiert, dazu eine lineare Probe auf den Lippenlese-Embeddings.

* Jeder Normalisierungsschritt sollte etwas Wortgenauigkeit kosten und etwas Identität entfernen. Beides zu messen ist das Kernergebnis.
* **Hauptabbildung:** Wortgenauigkeit gegen Sprecher-ID-Genauigkeit, ein Punkt pro Eingabe.

### 5. Neutraler-Mund-Anonymisierer
Die Sprecher a, b und c erscheinen alle als ein neutraler Mund d, der dieselben Bewegungen macht. Das ist das visuelle Gegenstück zur Umwandlung jeder Stimme in eine Zielstimme, einer Standardmethode der Sprach-Anonymisierung.

* **Encoder:** wandelt den Eingabemund in einen Bewegungscode um, der enthält, was der Mund tut (Öffnung, Rundung, Zähne, Zunge), aber nichts darüber, wessen Mund es ist.
* **Landmarken-Decoder (Kern):** gibt Lippenpunkte auf einer kanonischen Mundform aus.
* **RGB-Decoder (Erweiterung):** rendert Mund d als höher aufgelöstes Video.
* **Training:** Es gibt keine gepaarten Daten (Mund d hat nie dasselbe Wort gesagt), daher tragen drei Losses das Training:
    * **Inhalt:** Ein eingefrorener Lippenleser muss das Wort in der Ausgabe weiterhin erkennen.
    * **Identität:** Ein adversarialer Sprecher-Klassifikator auf dem Code, den der Encoder überlisten muss.
    * **Realismus:** ein Realismus-Loss für die RGB-Ausgabe.
* **Zu schlagende Baseline:** die mittlere Lippenform jedes Sprechers abziehen und die kanonische addieren.
* **Mund d:** ein gemittelter oder synthetischer Mund, nie eine reale Person.
* **Evaluation:** Wortgenauigkeit eines Lippenlesers auf der Ausgabe und Sprecher-ID durch einen Angreifer, der auf anonymisierten Ausgaben trainiert ist, denn der Bewegungsstil (wie weit, schnell oder asymmetrisch jemand spricht) kann Personen weiterhin identifizieren. Beide Varianten werden neue Punkte im Zielkonflikt-Plot aus Baustein 4.
* **Veröffentlichung:** Das RGB-Modell erzeugt Mundvideo, das aus Bundestagsmaterial gelernt wurde. Vor der Freigabe seiner Gewichte wird beim Bundestag nachgefragt.
