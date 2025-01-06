import numpy as np

def solve_riccati(A, B, Q, R, N):
    """
    Rozwiązuje równanie Riccatiego wstecz dla optymalizacji liniowo-kwadratowej (LQR).
    """
    P = Q  # Macierz Riccatiego na końcu horyzontu czasowego
    Ps = [P]
    for _ in range(N):
        P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        Ps.insert(0, P)
    return Ps

def compute_optimal_solution_fixed(N):
    """
    Oblicza optymalne sterowanie u(k) i wskaźnik jakości J analitycznie.
    """
    # Parametry systemu
    A = np.array([[0, 1], [-1, 2]])  # Macierz stanu
    B = np.array([[0], [1 / N**2]])  # Macierz sterowania
    Q = np.array([[1, 0], [0, 0]])  # Kara za stan x1 (maksymalizacja x1)
    R = np.array([[1]])             # Kara za sterowanie

    # Rozwiązanie równań Riccatiego
    Ps = solve_riccati(A, B, Q, R, N)

    # Symulacja optymalnego sterowania i stanu
    x = np.zeros((2, N+1))  # Stany: [x1, x2]
    u = np.zeros(N)         # Sterowania

    for k in range(N):
        P = Ps[k]
        K = -np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        u[k] = K @ x[:, k]  # Optymalne sterowanie
        x[:, k+1] = A @ x[:, k] + B.flatten() * u[k]

    # Obliczanie wskaźnika jakości J
    x1_N = x[0, -1]
    effort = np.sum(u**2)
    J = x1_N - (1 / (2 * N)) * effort

    return u, J, x

# Analiza dla różnych wartości N
results_analytical_fixed = {}
for N in [5, 10, 15, 20, 25, 30, 35, 40, 45]:
    u_optimal, J_optimal, x_states = compute_optimal_solution_fixed(N)
    results_analytical_fixed[N] = (u_optimal, J_optimal)
    print(f"N={N}, Optimal J={J_optimal}, Optimal u={u_optimal}")
