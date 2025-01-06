from scipy.optimize import minimize
import numpy as np

def simulate_system(u, N):
    x1 = np.zeros(N + 1)
    x2 = np.zeros(N + 1)

    for k in range(N):
        x1[k + 1] = x2[k]
        x2[k + 1] = 2 * x2[k] - x1[k] + (1 / N**2) * u[k]

    return x1, x2

# Funkcja celu dla optymalizacji
def objective(u, N):
    u = np.array(u)
    x1, x2 = simulate_system(u, N)
    effort_penalty = (1 / (2 * N)) * np.sum(u**2)
    return -(x1[-1] - effort_penalty)  # Negujemy, bo `minimize` minimalizuje, a my chcemy maksymalizować

# Ograniczenia na sterowanie u(k) (np. w przedziale [-1, 1])
def control_bounds(N):
    return [(-1, 1) for _ in range(N)]

# Znajdowanie rozwiązania optymalnego
def find_optimal_solution(N):
    # Inicjalna zgadywana sekwencja sterowań
    initial_guess = np.zeros(N)
    
    # Ograniczenia na sterowania
    bounds = control_bounds(N)
    
    # Rozwiązanie optymalizacyjne
    result = minimize(
        objective, 
        initial_guess, 
        args=(N,), 
        bounds=bounds, 
        method='SLSQP',  # Sekwencyjne programowanie kwadratowe
        options={'disp': False}
    )
    
    # Optymalna sekwencja sterowań i wartość funkcji celu
    optimal_u = result.x
    optimal_J = -result.fun  # Negujemy z powrotem na maksymalizację
    return optimal_u, optimal_J

# Porównanie dla przykładowych wartości N
Ns = [5, 10, 15, 20, 25, 30, 35, 40, 45]
optimal_results = {}

for N in Ns:
    print(f"Obliczanie rozwiązania optymalnego dla N = {N}...")
    optimal_u, optimal_J = find_optimal_solution(N)
    optimal_results[N] = (optimal_u, optimal_J)
    # print(f"Optymalne u = {optimal_u}")
    print(f"Optymalne J = {optimal_J}")
