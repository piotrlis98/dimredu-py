# Eksperyment z klasyfikacją i selekcją cech

Kod _main.py_ jest skryptem w języku Python, który wykonuje eksperyment z klasyfikacją i selekcją cech na różnych zbiorach danych.

## Kroki

### Importowanie pakietów

Kod zaczyna od importowania wymaganych pakietów.

### Wczytywanie danych

Kod wczytuje dane z wybranego zbioru danych spośród pięciu dostępnych opcji: breast_cancer, pistachio, sonar, weather i nba.

### Wybór metody i klasyfikatora

Następnie kod wybiera metodę redukcji wymiarowości lub selekcji cech spośród siedmiu dostępnych opcji: PCA, LDA, CA, Chi2, RFE, VT i SFS. Wybiera również klasyfikator spośród ośmiu dostępnych opcji: SVC, LogisticRegression, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier, XGBClassifier, MLPClassifier i RandomForestClassifier.

### Przygotowanie danych

Dane są dzielone na zbiór treningowy i testowy, a następnie są skalowane za pomocą MinMaxScaler.

### Redukcja wymiarowości i klasyfikacja

Metoda redukcji wymiarowości lub selekcji cech jest stosowana na zbiorze treningowym i testowym, a następnie klasyfikator jest dopasowywany na zbiorze treningowym.

### Optymalizacja

Parametry metody i klasyfikatora są optymalizowane za pomocą GridSearchCV.

### Predykcja i ocena

Kod przewiduje etykiety na zbiorze testowym i oblicza metryki klasyfikacji, takie jak precyzja, czułość, miara F1, dokładność i współczynnik korelacji Matthews'a.

### Powtórzenie eksperymentu

Eksperyment jest powtarzany określoną liczbę razy i obliczane są średnie metryki klasyfikacji dla każdej metody i klasyfikatora.

### Wizualizacja wyników

Kod tworzy i zapisuje heatmapę i wykres wartości własnych macierzy korelacji dla każdego zbioru danych. Tworzy również heatmapę metryk klasyfikacji dla każdego klasyfikatora.

### Zapis wyników

Wartości współczynnika korelacji Matthews'a dla każdej metody i klasyfikatora są zapisywane do pliku csv.
