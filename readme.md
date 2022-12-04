# Knobeln-Scanner

## Übersicht

1. MNIST Model mit Keras trainieren
2. Model speichern
3. opencv nutzen, um Tabelle einzulesen
4. In jeder entsprechenden Zelle einzelne Zahlen auslesen auf folgende Art:
    - Einzelne Zelle blurry und canny anwenden
    - findContours anwenden und Größe prüfen (min- und max-Werte festlegen)
    - Box auslesen und Zahl zentrieren für MNIST
5. Das entstandene Bild auf 32x32 (MNIST) resizen
6. Bild mit oben erstelltem MNIST-Model auslesen und Zahl erhalten.
7. Für jede Zahl wiederholen
8. Rechnen...
