import numpy as np
import plots


def shift_r(x):
    return np.concatenate((np.zeros_like(x[..., :, -1:]), x[..., :, :-1]), axis=-1)

def shift_l(x):
    return np.concatenate((x[..., :, 1:], np.zeros_like(x[..., :, :1])), axis=-1)

def shift_t(x):
    return np.concatenate((np.zeros_like(x[..., -1:, :]), x[..., :-1, :]), axis=-2)

def shift_b(x):
    return np.concatenate((x[..., 1:, :], np.zeros_like(x[..., :1, :])), axis=-2)

def evaluate(x, size=20):
    assert np.shape(x)[-1] == size * size
    grid = np.asarray(np.reshape(x, (-1, size, size)), dtype=np.int_)
    points = np.minimum(
        shift_r(grid) + shift_l(grid) + shift_t(grid) + shift_b(grid),
        1 - grid
    )
    return points.reshape(np.shape(x)).sum(-1), grid, points



def roulette_selection(population, quality, result_size):
    probs = quality / np.sum(quality)
    selection = np.random.choice(len(population), size=result_size, replace=True, p=probs)
    return population[selection]

def crossover(population, crossover_prob):
    new_population = []
    np.random.shuffle(population)
    length = len(population[0])
    for i in range(0, len(population), 2):
        if np.random.rand() < crossover_prob:
            cross_point = np.random.randint(2, length)
            new_population.append(np.concatenate((population[i][:cross_point], population[i + 1][cross_point:])))
            new_population.append(np.concatenate((population[i + 1][:cross_point], population[i][cross_point:])))
        else:
            new_population.append(population[i])
            new_population.append(population[i + 1])
    new_population = np.array(new_population)
    return np.array(new_population)

def mutation(population, mutation_prob):
    for i in range(len(population)):
        for j in range(len(population[i])):
            if np.random.rand() < mutation_prob:
                population[i][j] = 1 - population[i][j]
    return population

def evolutionary_algorithm(population_size, mutation_prob, crossover_prob, num_generations, size=20, quality_func=evaluate):
    """
    pop_size - rozmiar populacji
    mutation_prob - prawdopodobieństwo mutacji
    recombination_prob - prawdopodobieństwo rekombinacji
    num_generations - liczba generacji
    size — rozmiar siatki
    quality_func - funkcja oceny
    """

    if population_size % 2 != 0:
        population_size += 1
    population = np.random.randint(0, 2, (population_size, size*size))
    #print(population)
    quality = quality_func(population, size)[0]
    best_solution = population[np.argmax(quality)]
    best_score = np.max(quality)

    for i in range(num_generations):
        if i % 100 == 0:
            print(f'Iteracja {i}/{num_generations}, najlepszy wynik: {best_score}')
        new_population = roulette_selection(population, quality, population_size)
        new_population = crossover(new_population, crossover_prob)
        new_population = mutation(new_population, mutation_prob)

        quality = quality_func(new_population, size)[0]
        best_new_solution = new_population[np.argmax(quality)]
        best_new_score = np.max(quality)
        if best_new_score > best_score:
            best_solution = best_new_solution
            best_score = best_new_score

        population = new_population
    return best_solution, best_score







results = evolutionary_algorithm(population_size=800, mutation_prob=0.01, crossover_prob=0.8, num_generations=10000, size = 20)
print(results[1])
score, grid, points = evaluate(results[0], 20)

plots.visualize(grid, points, 20)






# size = 20  # Define grid size
# x = np.random.randint(0, 2, size * size)  # Random binary grid
# print(np.shape(x)[-1])
# print(x)
# score, grid, points = evaluate(x, size)
# print(score)
# print(grid)
# print(points)

