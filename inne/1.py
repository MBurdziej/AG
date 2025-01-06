import numpy as np

# Parametry algorytmu genetycznego
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

def simulate_cart(N, u):

    # Symulacja układu wózka dla zadanych sterowań u.

    x1, x2 = 0, 0
    for k in range(N):
        x1, x2 = x2, 2 * x2 - x1 + (1 / N**2) * u[k]
    return x1

def calculate_fitness(N, population):

    # Funkcja celu do oceny jakości każdego osobnika w populacji.

    fitness = []
    for u in population:
        u = np.array(u)
        x1_N = simulate_cart(N, u)
        effort = np.sum(u**2)
        J = x1_N - (1 / (2 * N)) * effort
        fitness.append(J)
    return np.array(fitness)

def select_parents(population, fitness):

    # Wybór rodziców metodą ruletki.

    probabilities = fitness / np.sum(fitness)
    indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[indices[0]], population[indices[1]]

def crossover(parent1, parent2):

    # Operacja krzyżowania jednopunktowego.

    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual, mutation_rate):

    # Operacja mutacji.

    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 1)
    return individual

def genetic_algorithm(N):

    # Główna funkcja realizująca algorytm genetyczny dla zadania.

    # Inicjalizacja populacji
    population = [np.random.uniform(-5, 5, N) for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        # Obliczanie jakości populacji
        fitness = calculate_fitness(N, population)
        
        # Tworzenie nowej populacji
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, MUTATION_RATE)
            child2 = mutate(child2, MUTATION_RATE)
            new_population.append(child1)
            new_population.append(child2)
        
        population = new_population

    # Obliczenie najlepszego rozwiązania
    fitness = calculate_fitness(N, population)
    best_index = np.argmax(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]

    return best_solution, best_fitness

# Rozwiązanie dla różnych wartości N
results = {}
for N in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
    solution, fitness = genetic_algorithm(N)
    results[N] = (solution, fitness)
    print(f"N={N}, Best Fitness={fitness}, Best Solution={solution}")
