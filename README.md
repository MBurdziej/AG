# Algorytm Genetyczny do Problemu Pchania Wózka

## Opis

Ten projekt implementuje algorytm genetyczny do rozwiązania problemu sterowania ruchem wózka, który jest sformułowany jako problem optymalizacji trajektorii w zadanym czasie. Celem jest maksymalizacja całkowitej drogi przebytej przez wózek przy jednoczesnej minimalizacji wysiłku sterowania.

Problem jest modelowany dyskretnie według równań:

```
x1(k + 1) = x2(k)
x2(k + 1) = 2 * x2(k) - x1(k) + (1 / N^2) * u(k)
```

gdzie `x1` to pozycja, `x2` to prędkość, a `u(k)` to sterowanie.

Funkcja celu, którą maksymalizuje algorytm genetyczny, ma postać:

```
J = x1(N) - (1 / (2 * N)) * SUMA(u^2)
```

czyli końcowa pozycja minus kara za użycie siły sterowania.

## Wymagania

- Python 3.7+
- NumPy
- Matplotlib

## Instalacja

1. Zainstaluj wymagane biblioteki:
   ```sh
   pip install numpy matplotlib
   ```

## Struktura kodu

- **`simulate_system(u, N)`** – symuluje dynamikę systemu dla zadanej sekwencji sterowania.
- **`evaluate_fitness(u, N)`** – oblicza wskaźnik jakości `J` dla danej trajektorii.
- **`initialize_population(population_size, N)`** – tworzy początkową populację rozwiązań.
- **`tournament_selection(population, fitness, tournament_size, num_selected)`** – wybiera najlepszych osobników metodą turniejową.
- **`crossover(selected, population_size, N)`** – stosuje krzyżowanie jednopunktowe do generowania nowej populacji.
- **`mutate(population, mutation_rate, N)`** – wprowadza mutacje do populacji.
- **`genetic_algorithm(N, population_size, generations, mutation_rate)`** – główny algorytm genetyczny iterujący przez generacje.

## Uruchomienie

Skrypt automatycznie uruchamia optymalizację dla różnych wartości `N`:

```sh
python AG_pchanie_wozka.py
```

## Wynik

Dla każdego `N` program wypisuje najlepsze rozwiązanie i rysuje wykresy:

- **Wskaźnik J w zależności od N** (`J(N).png`)
- **Najlepsze sterowanie u dla różnych N** (`u(N).png`)
- **Trajektorie ruchu wózka dla różnych N** (`x1(N).png`)
- **Poprawa wskaźnika J w trakcie iteracji** (`J_dla_N_X.png` dla różnych `N`)
