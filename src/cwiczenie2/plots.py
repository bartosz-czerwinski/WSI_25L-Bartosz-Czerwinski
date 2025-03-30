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

def plot_population_results(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_pop = None
    current_scores = []

    for line in lines:
        line = line.strip()
        if line.startswith("Population size"):
            if current_pop is not None and current_scores:
                save_plot1(current_pop, current_scores)
            current_pop = int(line.split(":")[1].strip())
            current_scores = []
        elif line:
            current_scores = list(map(int, line.split(",")))

    # Zapisz ostatni wykres
    if current_pop is not None and current_scores:
        save_plot1(current_pop, current_scores)


def save_plot1(population_size, scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label=f"Wielkość populacji {population_size}")
    plt.xlabel("Czas działania algorytmu [s]")
    plt.ylabel("Najlepszy wynik")
    plt.title(f"Przebieg działania algorytmu, wielkość populacji: {population_size}")
    plt.grid(True)
    plt.legend()
    filename = f"popsize_{population_size}.png"
    plt.savefig(filename)
    plt.close()


def plot_summary_final_scores(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    pop_sizes = []
    final_scores = []
    current_pop = None
    current_scores = []

    for line in lines:
        line = line.strip()
        if line.startswith("Population size"):
            if current_pop is not None and current_scores:
                pop_sizes.append(current_pop)
                final_scores.append(current_scores[-1])
            current_pop = int(line.split(":")[1].strip())
            current_scores = []
        elif line:
            current_scores = list(map(int, line.split(",")))

    if current_pop is not None and current_scores:
        pop_sizes.append(current_pop)
        final_scores.append(current_scores[-1])

    plt.figure(figsize=(10, 6))
    plt.plot(pop_sizes, final_scores, marker='o')
    plt.xlabel("Wielkość populacji")
    plt.ylabel("Końcowy najlepszy wynik")
    plt.title("Wynik algorytmu w zależności od wielkości populacji")
    plt.grid(True)
    plt.savefig("summary_population_vs_score.png")
    plt.close()
    print("Zapisano wykres zbiorczy: summary_population_vs_score.png")


