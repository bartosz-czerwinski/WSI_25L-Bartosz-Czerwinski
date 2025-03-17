import numpy as np
import matplotlib
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import title

matplotlib.use('TkAgg')


def plot_function(f, x_range=(-10, 10), y_range=(-10, 10), resolution=1000, result_range=None, filename="Plot"):
    """
    Rysuje wykres dowolnej funkcji f.

    - Jeśli f przyjmuje jeden argument: wykres 2D
    - Jeśli f przyjmuje dwa argumenty: wykres 3D

    Parametry:
    f - funkcja do narysowania
    x_range - zakres wartości x (dla 2D i 3D)
    y_range - zakres wartości y (dla 3D)
    result_range - zakres wartości y (dla 2D) oraz z dla (3D) (opcjonalnie)
    resolution - ilość punktów
    """
    num_args = f.__code__.co_argcount
    if num_args == 1:  # Wykres 2D
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.vectorize(f)(x)  # Obliczanie wartości funkcji

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=f"f(x)", color="blue")

        # Oznaczenia osi
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(True, linestyle='--', linewidth=0.5)

        # Etykiety i legenda
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Wykres funkcji " + f.__name__ + "(x)")
        plt.legend()

        # Ustawienie zakresu osi y, jeśli podano
        if result_range:
            plt.ylim(result_range)

        plt.savefig(filename + ".png")

    if num_args == 2:  # Wykres 3D
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(f)(X, Y)  # Obliczanie wartości funkcji
        if result_range:
            for i in range(len(Z)):
                for j in range(len(Z[i])):
                    if Z[i][j] > result_range[1] or Z[i][j] < result_range[0]:
                        Z[i][j] = None

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='ocean')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(f.__name__ + "(x1, x2)")
        ax.set_title("Wykres funkcji " + f.__name__ + "(x1, x2)")

        # Ustawienie zakresu osi z, jeśli podano
        if result_range:
            ax.set_zlim(result_range)

        plt.savefig(filename + ".png")


def create_gif(f, results, x_range=(-5, 5), y_range=(-5, 5), result_range=None, resolution=100,
               filename="gradient_descent", title="Metoda gradientu prostego"):
    if results is False:
        print("Obliczenia zostały przerwane z powodu zbyt dużego gradientu.")
        return
    x_min, trajectory = results
    # Zmiana trajectory tak aby na gifie znajdowalo sie tylko 100 krokow
    if len(trajectory) > 100:
        trajectory = trajectory[::len(trajectory) // 100]
    for i in range(3):
        trajectory = np.vstack((trajectory[0], trajectory))
    num_args = f.__code__.co_argcount

    if num_args == 1:
        # Tworzenie animacji 2D
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = f(x)
        ax.plot(x, y, label="f(x)", color='blue')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel(f.__name__ + "(x)")
        ax.set_title(title)
        point, = ax.plot([], [], 'ro', markersize=6)
        path, = ax.plot([], [], 'r-', linewidth=1)

        if result_range:
            plt.ylim(result_range)

        def update(frame):
            x_vals = trajectory[:frame + 1, 0]
            y_vals = f(x_vals)
            point.set_data([x_vals[-1]], [y_vals[-1]])
            path.set_data(x_vals, y_vals)
            return point, path

        filename1 = filename + ".gif"

        ani = animation.FuncAnimation(fig, update, frames=len(trajectory), repeat=True)
        ani.save(filename1, writer="pillow", fps=5)
        plt.close(fig)

    elif num_args == 2:
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(f)(X, Y)

        # Tworzenie animacji 3D
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        ax_3d.plot_surface(X, Y, Z, cmap='ocean', alpha=0.6)
        ax_3d.set_xlabel("x1")
        ax_3d.set_ylabel("x2")
        ax_3d.set_zlabel(f.__name__ + "(x1, x2)")
        ax_3d.set_title(title + " 3D")
        point_3d, = ax_3d.plot([], [], [], 'ro', markersize=6)
        path_3d, = ax_3d.plot([], [], [], 'r-', linewidth=1)

        def update_3d(frame):
            if frame % 10 == 0 and frame != 0:
                print(f"Generowanie klatki {frame}/{len(trajectory)}")
            x_vals, y_vals = trajectory[:frame + 1, 0], trajectory[:frame + 1, 1]
            z_vals = np.array([f(x, y) for x, y in zip(x_vals, y_vals)])
            point_3d.set_data([x_vals[-1]], [y_vals[-1]])
            point_3d.set_3d_properties([z_vals[-1]])
            path_3d.set_data(x_vals, y_vals)
            path_3d.set_3d_properties(z_vals)
            return point_3d, path_3d

        filename1 = filename + "_3D.gif"

        ani_3d = animation.FuncAnimation(fig_3d, update_3d, frames=len(trajectory), repeat=True)
        ani_3d.save(filename1, writer="pillow", fps=5)
        plt.close(fig_3d)

        # Tworzenie animacji konturowej (warstwicowej)
        fig_contour = plt.figure(figsize=(8, 6))
        ax_contour = fig_contour.add_subplot(111)
        ax_contour.contourf(X, Y, Z, levels=20, cmap='ocean')
        ax_contour.set_xlabel("x")
        ax_contour.set_ylabel("y")
        ax_contour.set_title(title + " poziomica")
        point_contour, = ax_contour.plot([], [], 'ro', markersize=6)
        path_contour, = ax_contour.plot([], [], 'r-', linewidth=1)

        def update_contour(frame):
            if frame % 10 == 0 and frame != 0:
                print(f"Generowanie klatki {frame}/{len(trajectory)}")
            x_vals, y_vals = trajectory[:frame + 1, 0], trajectory[:frame + 1, 1]
            point_contour.set_data([x_vals[-1]], [y_vals[-1]])
            path_contour.set_data(x_vals, y_vals)
            return point_contour, path_contour

        filename1 = filename + "_poziomica.gif"
        ani_contour = animation.FuncAnimation(fig_contour, update_contour, frames=len(trajectory), repeat=True)
        ani_contour.save(filename1, writer="pillow", fps=5)
        plt.close(fig_contour)

        print("GIFy wygenerowane")
    else:
        print("Można narysować tylko funkcje jednego lub dwóch argumentów.")


def plot_avg_iterations(avg_iterations, std_iterations, function_name, title="Średnia liczba iteracji"):
    """
    Rysuje wykres średniej liczby iteracji do minimum w zależności od `learning_rate`,
    z uwzględnieniem odchylenia standardowego.
    """
    plt.figure(figsize=(8, 5))
    lr_values = list(avg_iterations.keys())
    avg_values = list(avg_iterations.values())
    std_values = list(std_iterations.values())

    plt.errorbar(lr_values, avg_values, yerr=std_values, fmt='o-', capsize=5, label="Średnia liczba iteracji")

    plt.xlabel("Krok")
    plt.ylabel("Średnia liczba iteracji do minimum")
    plt.title(f"Średnia liczba iteracji dla {function_name}")
    plt.xscale("log")  # Skala logarytmiczna
    plt.grid()
    plt.legend()
    plt.savefig(f"{title}.png", dpi=600, bbox_inches="tight")


def plot_correct_ratio(correct_ratio, function_name, title="Skuteczność metody"):
    """
    Rysuje wykres przedstawiający skuteczność metody dla różnych wartości kroku.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list(correct_ratio.keys()), list(correct_ratio.values()), marker='o', linestyle='-', color="green")
    # nazwy osi po polsku
    plt.xlabel("Krok")
    plt.ylabel("Skuteczność metody")
    plt.title(f"Skuteczność metody gradientowej dla {function_name}")
    plt.xscale("log")  # Skala logarytmiczna
    plt.ylim(0, 1.1)
    plt.grid()
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
