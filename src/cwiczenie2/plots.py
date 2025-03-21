import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
matplotlib.use('TkAgg')

def visualize(grid, points, size=20):
    grid = grid[0]
    points = points[0]
    colors = ["yellow", "lightgreen", "darkgreen"]  # 0: No points, 1: Point, 2: Object
    cmap = mcolors.ListedColormap(colors)

    display_grid = np.zeros_like(grid, dtype=int)
    display_grid[grid == 1] = 2  # Objects
    display_grid[(grid == 0) & (points == 1)] = 1  # Adjacent to objects

    plt.figure(figsize=(6, 6))
    plt.imshow(display_grid, cmap=cmap, origin='upper')
    plt.xticks(range(size), range(1, size + 1))
    plt.yticks(range(size), range(1, size + 1))
    plt.grid(False)
    plt.show()