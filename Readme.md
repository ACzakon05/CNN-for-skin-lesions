# Image Classifier Project (CPU-friendly)

Projekt klasyfikuje obrazy do 3 klas: healthy,melanoma, melanocytic nevi, używając EfficientNetB0.

## Struktura katalogów
- data/train/, data/validate/, data/test/ – obrazy posegregowane po klasach
- src/ – kod źródłowy
- requirements.txt – potrzebne biblioteki

## Uruchamianie

1. Zainstaluj środowisko:
   pip install -r requirements.txt

2. Trening (tylko głowa modelu, CPU-friendly):
   python -m src.train

3. Ewaluacja:
   python -m src.evaluate

4. Predict:
    python -m src.predict
    ścieżka

## Parametry
batch_size = 8 # zwiększyć do 16 albo 32
epochs_head = 2 # zwiększyć do minimalnie 15 jeśli GPU
epochs_fine = 1 # zwiększyć do minimalnie 10 jeśli GPU
zwiększyć rozdzielczość zdjęcia (320,320) 

## Uwagi
- Fine-tuning base_model zakomentowany, odblokować tylko jeśli masz GPU.
- Włączony oversampling i włączona waga dla melanoma ( bardziej poprawne lekarsko )

## Nowe klasy:
- Dodaj do data nowe klasu
-w src/trin.py : model=create_model(num_classes=<NEW>)
-run training
-run evaluation
