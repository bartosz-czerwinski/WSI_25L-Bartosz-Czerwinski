import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

def visualize(grid, points, score, size=20):
    grid = grid[0]
    points = points[0]
    colors = ["yellow", "lightgreen", "darkgreen"]  # 0: No points, 1: Point, 2: Object
    cmap = mcolors.ListedColormap(colors)

    display_grid = np.zeros_like(grid, dtype=int)
    display_grid[grid == 1] = 2  # Objects
    display_grid[(grid == 0) & (points == 1)] = 1  # Adjacent to objects

    plt.figure(figsize=(6, 6))
    plt.imshow(display_grid, cmap=cmap, origin='upper', vmin=0, vmax=2)
    plt.xticks(range(size), range(1, size + 1))
    plt.yticks(range(size), range(1, size + 1))
    plt.grid(False)
    plt.title(f"Wynik: {score}")
    plt.show()


def plot_data_from_file(filename, plot_type):

    # Odczytaj dane z pliku
    with open(filename, 'r') as file:
        data = file.read()

    # Wyodrębnij dane z tekstu
    population_sizes = []
    best_scores_population = []
    mutation_probabilities = []
    best_scores_mutation = []
    crossover_probabilities = []
    best_scores_crossover = []

    # Użyj wyrażenia regularnego do znalezienia wielkości populacji, prawdopodobieństwa mutacji i najlepszych wyników
    matches_population = re.findall(r"Population size: (\d+)\nBest score: (\d+)", data)
    for size, score in matches_population:
        population_sizes.append(int(size))
        best_scores_population.append(int(score))

    matches_mutation = re.findall(r"Mutation prob: (\d+\.\d+)\nBest score: (\d+)", data)
    for prob, score in matches_mutation:
        mutation_probabilities.append(float(prob))
        best_scores_mutation.append(int(score))

    matches_crossover = re.findall(r"Crossover prob: (\d+\.\d+)\nBest score: (\d+)", data)
    for prob, score in matches_crossover:
        crossover_probabilities.append(float(prob))
        best_scores_crossover.append(int(score))

    # Wygeneruj wykres w zależności od typu
    if plot_type == 'population':
        plt.figure(figsize=(10, 5))
        plt.plot(population_sizes, best_scores_population, marker='o')
        plt.xlabel("Wielkość Populacji")
        plt.ylabel("Najlepszy Wynik")
        plt.title("Zależność Najlepszego Wyniku od Wielkości Populacji")
        plt.grid(True)
        plt.savefig('population_plot.png')
    elif plot_type == 'mutation':
        plt.figure(figsize=(10, 5))
        plt.plot(mutation_probabilities, best_scores_mutation, marker='o')
        plt.xlabel("Prawdopodobieństwo Mutacji")
        plt.ylabel("Najlepszy Wynik")
        plt.title("Zależność Najlepszego Wyniku od Prawdopodobieństwa Mutacji")
        plt.grid(True)
        plt.savefig('mutation_plot.png')
    elif plot_type == 'crossover':
        plt.figure(figsize=(10, 5))
        plt.plot(crossover_probabilities, best_scores_crossover, marker='o')
        plt.xlabel("Prawdopodobieństwo Krzyżowania")
        plt.ylabel("Najlepszy Wynik")
        plt.title("Zależność Najlepszego Wyniku od Prawdopodobieństwa Krzyżowania")
        plt.grid(True)
        plt.savefig('crossover_plot.png')
    else:
        print("Nieznany typ wykresu.")
        return

    plt.grid(True)
    plt.show()
