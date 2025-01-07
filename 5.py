import numpy as np
import matplotlib.pyplot as plt

# Funkcja modelu systemu definuje zmiany stanów x1, x2
def simulate_system(u, N):
    x1 = np.zeros(N + 1)
    x2 = np.zeros(N + 1)

    for k in range(N):
        x1[k + 1] = x2[k]
        x2[k + 1] = 2 * x2[k] - x1[k] + (1 / N**2) * u[k]

    return x1, x2

# Funkcja celu
# Oblicza wskaźnik J na podstawie trajektorii x1 i sterowania u
def evaluate_fitness(u, N):
    x1, _ = simulate_system(u, N)
    effort_penalty = (1 / (2 * N)) * np.sum(u**2)
    return x1[-1] - effort_penalty

# Inicjalizacja populacji
def initialize_population(population_size, N):
    return np.random.uniform(-1, 1, (population_size, N))

# Selekcja turniejowa
def tournament_selection(population, fitness, tournament_size, num_selected):
    selected = []
    for _ in range(num_selected):
        tournament_indices = np.random.choice(range(len(population)), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_index])
    return np.array(selected)

# Krzyżowanie jednopunktowe
def crossover(selected, population_size, N):
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]
        crossover_point = np.random.randint(1, N - 1)
        offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        new_population.append(offspring)
    return np.array(new_population)

# Mutacja
def mutate(population, mutation_rate, N):
    mutations = np.random.uniform(-1, 1, population.shape)
    mutation_mask = np.random.rand(*population.shape) < mutation_rate
    population[mutation_mask] += mutations[mutation_mask]
    return np.clip(population, -1, 1)  # Sterowanie ograniczone do [-1, 1]

# Algorytm genetyczny
def genetic_algorithm(N, population_size=1000, generations=500, mutation_rate=0.005):
    # Inicjalizacja populacji
    population = initialize_population(population_size, N)
    best_fitness_per_generation = []
    average_fitness_per_generation = []

    # Ewolucja populacji przez zadane liczby generacji
    for generation in range(generations):
        # Obliczanie dopasowania dla każdego osobnika w populacji
        fitness = np.array([evaluate_fitness(individual, N) for individual in population])

        # Selekcja turniejowa
        selected = tournament_selection(population, fitness, tournament_size=5, num_selected=population_size // 2)

        # Tworzenie nowej populacji przez krzyżowanie
        population = crossover(selected, population_size, N)

        # Mutacja
        population = mutate(population, mutation_rate, N)

        # Zapisywanie najlepszego i średniego wyniku w bieżącej generacji
        best_fitness_per_generation.append(np.max(fitness))
        average_fitness_per_generation.append(np.mean(fitness))

    # Zwrócenie najlepszego osobnika, jego wartości dopasowania i historii
    best_index = np.argmax([evaluate_fitness(individual, N) for individual in population])
    best_individual = population[best_index]
    best_fitness = evaluate_fitness(best_individual, N)

    return best_individual, best_fitness, best_fitness_per_generation, average_fitness_per_generation

# Główna pętla dla różnych wartości N
Ns = [5, 10, 15, 20, 25, 30, 35, 40, 45]
results = {}

for N in Ns:
    print(f"Optymalizacja dla N = {N}...")
    best_u, best_fitness, fitness_history, avg_fitness_history = genetic_algorithm(N)
    results[N] = (best_u, best_fitness, fitness_history, avg_fitness_history)
    print(f"Najlepszy wskaźnik J = {best_fitness}\n")

# Wizualizacja wskaźnika J dla różnych wartości N
Ns = list(results.keys())
fitness_values = [results[N][1] for N in Ns]

plt.figure(figsize=(12, 8))
plt.plot(Ns, fitness_values, marker='o', linestyle='-', label="Najlepszy wskaźnik J")
plt.title("Wskaźnik J w zależności od N")
plt.xlabel("N (liczba kroków)")
plt.ylabel("Najlepszy wskaźnik J")
plt.grid()
plt.legend()
plt.savefig('J(N).png', format='png', dpi=300)
plt.show()

# Wizualizacja sterowania u dla różnych N
plt.figure(figsize=(12, 8))
for N in Ns:
    best_u, _, _, _ = results[N]
    plt.plot(range(N), best_u, label=f"N={N}")
plt.title("Sterowanie u dla różnych wartości N")
plt.xlabel("Czas k")
plt.ylabel("u (sterowanie)")
plt.legend()
plt.grid()
plt.savefig('u(N).png', format='png', dpi=300)
plt.show()

# Wizualizacja trajektorii dla kilku N
plt.figure(figsize=(12, 8))
for N in Ns:  # Wybieramy reprezentatywne wartości N
    best_u, _, _, _ = results[N]
    x1, _ = simulate_system(best_u, N)
    plt.plot(range(N + 1), x1, label=f"N={N}")

plt.title("Trajektoria położenia x1 dla różnych wartości N")
plt.xlabel("Czas k")
plt.ylabel("x1 (położenie)")
plt.legend()
plt.grid()
plt.savefig('x1(N).png', format='png', dpi=300)
plt.show()

# Wizualizacja poprawy wskaźnika J w trakcie iteracji dla wybranych N

for N in Ns:  # Wybieramy reprezentatywne wartości N
    _, _, fitness_history, avg_fitness_history = results[N]
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(fitness_history)), fitness_history, label=f"Najlepszy J dla N={N}")
    plt.plot(range(len(avg_fitness_history)), avg_fitness_history, linestyle='--', label=f"Średnie J dla N={N}")

    plt.title(f"Poprawa wskaźnika J w trakcie iteracji dla N = {N}")
    plt.xlabel("Pokolenie")
    plt.ylabel("Wskaźnik J")
    plt.legend()
    plt.grid()
    plt.savefig(f'J_dla_N_{N}.png', format='png', dpi=300)
    plt.show()
