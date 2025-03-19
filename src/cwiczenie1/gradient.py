import numpy as np
import autograd.numpy as anp
from autograd import grad
import plots


# Definicje funkcji
def f(x):
    return 10 * x ** 4 + 3 * x ** 3 - 30 * x ** 2 + 10 * x


def g(x1, x2):
    return (x1 - 2) ** 4 + (x2 + 3) ** 4 + 2 * ((x1 - 2) ** 2) * (x2 + 3) ** 2


def find_gradient(function):
    """
    Oblicza gradient funkcji f przy pomocy automatycznego różniczkowania.
    """

    def unpacked_f(x):
        return function(*x)

    return grad(unpacked_f)


def gradient_descent(function, x0, learning_rate=0.001, tol=1e-3, max_iters=1000):
    """
    Algorytm optymalizacji metodą gradientu prostego.
    function - funkcja do optymalizacji
    x0 - punkt początkowy
    learning_rate - wielkość kroku
    tol - tolerancja (minimalna norma gradientu)
    max_iters - maksymalna liczba iteracji
    """
    x = anp.array(x0, dtype=float)
    trajectory = [x.copy()]
    grad_function = find_gradient(function)
    for i in range(max_iters):
        grad_val = grad_function(x)
        # Przerwanie obliczeń, jeśli gradient jest zbyt duży
        if np.linalg.norm(grad_val) > 1e50:
            return False
        # Przerwanie obliczeń, jeśli gradient jest mały (bardzo blisko minimum)
        if np.linalg.norm(grad_val) < tol:
            break
        x -= learning_rate * grad_val
        trajectory.append(x.copy())

    return x, np.array(trajectory)



def test_learning_rates(function, dim, learning_rates, num_tests=10, tol=1e-3, max_iters=1000):
    """
    Testuje wpływ różnych wartości `learning_rate` na liczbę iteracji do znalezienia minimum.

    function - funkcja do testowania
    dim - liczba zmiennych funkcji (1D lub 2D)
    learning_rates - lista wartości współczynnika uczenia
    num_tests - ile różnych losowych punktów testujemy dla każdej wartości `learning_rate`
    max_iters - maksymalna liczba iteracji
    """

    def random_initial_points(dim, num_points, range_min=-5, range_max=5):
        """
        Generuje losowe punkty początkowe w zadanym zakresie.
        """
        return np.random.uniform(range_min, range_max, (num_points, dim))

    avg_iterations = {}
    std_iterations = {}  # Odchylenie standardowe liczby iteracji
    correct_ratio = {}

    for lr in learning_rates:
        iteration_counts = []
        correct = 0
        x = random_initial_points(dim, num_tests)
        tests = num_tests
        print(lr)

        for i in range(num_tests):
            if (i % 10 == 0):
                print(f'{i}/{num_tests}')
            x0 = x[i]
            result = gradient_descent(function, x0, learning_rate=lr, tol=tol, max_iters=max_iters)

            if result:
                final_x, trajectory = result
                iteration_counts.append(len(trajectory))
                correct += 1
            else:
                tests -= 1

        # Obliczamy średnią i odchylenie standardowe liczby iteracji
        if tests == 0:
            avg_iterations[lr] = max_iters
            std_iterations[lr] = 0
        else:
            avg_iterations[lr] = np.mean(iteration_counts)
            std_iterations[lr] = np.std(iteration_counts, ddof=1)

        print(iteration_counts)

        correct_ratio[lr] = correct / num_tests  # Stosunek poprawnych minimizacji

    return avg_iterations, std_iterations, correct_ratio


def plots_and_gifs():
    plots.plot_function(f, x_range=(-5, 5), filename="Wykres_funkcji_f(x)_(1)")
    plots.plot_function(f, x_range=(-5, 5), result_range=(-50, 50), filename="Wykres_funkcji_f(x)_(2)")
    plots.plot_function(g, x_range=(-3, 7), y_range=(-8, 2), filename="Wykres_funkcji_g(x1,x2)")

    results_f_correct = gradient_descent(f, x0=[-3], learning_rate=0.0038, tol=1e-3, max_iters=10000)
    plots.create_gif(f, results_f_correct, x_range=(-3, 3), filename="Funkcja_f(x)_poprawne_minimum",
                     title="Metoda gradientu prostego dla funkcji f(x) - minimum globalne")

    results_f_wrong = gradient_descent(f, x0=[-3], learning_rate=0.004, tol=1e-3, max_iters=10000)
    plots.create_gif(f, results_f_wrong, x_range=(-3, 3), filename="Funkcja_f(x)_bledne_minimum",
                     title="Metoda gradientu prostego dla funkcji f(x) - minimum lokalne")

    results_g = gradient_descent(g, x0=[-2, 1], learning_rate=0.001, tol=1e-3, max_iters=10000)
    plots.create_gif(g, results_g, x_range=(-3, 7), y_range=(-8, 2), filename="Funkcja_g(x1,x2)_poprawnie",
                     title="Metoda gradientu prostego dla funkcji g(x1, x2)")


def plot_tests():
    learning_rates = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]

    # Testowanie dla funkcji f(x)
    avg_iterations_f, std_iterations_f, correct_ratio_f = test_learning_rates(f, dim=1, learning_rates=learning_rates,
                                                                              num_tests=50, max_iters=2000)

    plots.plot_avg_iterations(avg_iterations_f, std_iterations_f, "f(x)",
                              title="Srednia_liczba_iteracji_w_zaleznosci_od_wielkosci_kroku_f(x)")
    plots.plot_correct_ratio(correct_ratio_f, "f(x)",
                             title="Skutecznosc_znajdowania_minimum_w_zaleznosci_od_wielkosci_kroku_f(x)")

    # Testowanie dla funkcji g(x, y)
    avg_iterations_g, std_iterations_g, correct_ratio_g = test_learning_rates(g, dim=2, learning_rates=learning_rates,
                                                                              num_tests=50, max_iters=5000)

    print(avg_iterations_g)
    print(correct_ratio_g)

    plots.plot_avg_iterations(avg_iterations_g, std_iterations_g, "g(x1, x2)",
                              title="Srednia_liczba_iteracji_w_zaleznosci_od_wielkosci_kroku_g(x1,x2)")
    plots.plot_correct_ratio(correct_ratio_g, "g(x1, x2)",
                             title="Skutecznosc_znajdowania_minimum_w_zaleznosci_od_wielkosci_kroku_g(x1,x2)")


plots_and_gifs()
plot_tests()
