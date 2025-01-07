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

# Algorytm genetyczny
def genetic_algorithm(N, population_size=100, generations=500, mutation_rate=0.05):
    # Inicjalizacja populacji (losowe sterowania u)
    population = np.random.uniform(-1, 1, (population_size, N))

    # Ewolucja populacji przez zadane liczby generacji
    for generation in range(generations):
        # Obliczanie dopasowania dla każdego osobnika w populacji
        fitness = np.array([evaluate_fitness(individual, N) for individual in population])

        # Selekcja najlepszych osobników (turniejowa lub proporcjonalna)
        selected_indices = np.argsort(fitness)[-population_size // 2:]
        selected = population[selected_indices]

        # Tworzenie nowej populacji przez krzyżowanie
        new_population = []
        while len(new_population) < population_size:
            # Losowy dobór rodziców
            parent1, parent2 = selected[np.random.choice(len(selected), 2, replace=False)]

            # Jednopunktowe krzyżowanie
            crossover_point = np.random.randint(1, N - 1)
            offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            new_population.append(offspring)

        new_population = np.array(new_population)

        # Mutacja (z małą szansą zmieniamy każdą wartość u)
        mutations = np.random.uniform(-1, 1, new_population.shape)
        mutation_mask = np.random.rand(*new_population.shape) < mutation_rate
        new_population[mutation_mask] += mutations[mutation_mask]

        # Aktualizacja populacji
        population = np.clip(new_population, -1, 1)  # Sterowanie ograniczone do [-1, 1]

    # Zwrócenie najlepszego osobnika i jego wartości dopasowania
    best_index = np.argmax([evaluate_fitness(individual, N) for individual in population])
    best_individual = population[best_index]
    best_fitness = evaluate_fitness(best_individual, N)

    return best_individual, best_fitness

# Główna pętla dla różnych wartości N
Ns = [5, 10, 15, 20, 25, 30, 35, 40, 45]
results = {}

for N in Ns:
    print(f"Optymalizacja dla N = {N}...")
    best_u, best_fitness = genetic_algorithm(N)
    results[N] = (best_u, best_fitness)
    print(f"Najlepszy wskaźnik J = {best_fitness:.4f}\n")

# Wizualizacja wyników dla jednego z przypadków (np. N=20)
N_visualize = 20
best_u, _ = results[N_visualize]
x1, x2 = simulate_system(best_u, N_visualize)

plt.figure()
plt.plot(range(N_visualize + 1), x1, label="x1 (położenie)")
plt.plot(range(N_visualize + 1), x2, label="x2 (prędkość)")
plt.title(f"Trajektoria dla N = {N_visualize}")
plt.xlabel("Czas k")
plt.ylabel("Wartość")
plt.legend()
plt.savefig('wykres.png', format='png', dpi=1000)
plt.show()
