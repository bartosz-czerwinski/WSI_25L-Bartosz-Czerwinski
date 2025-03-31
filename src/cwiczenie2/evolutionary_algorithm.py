import numpy as np
import plots
import time


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
        mask = np.random.rand(len(population[i])) < mutation_prob
        population[i][mask] = 1 - population[i][mask]
    return population


def evolutionary_algorithm(population_size, mutation_prob, crossover_prob, FES=1_000_000, size=20,
                           quality_func=evaluate):
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
    population = np.random.randint(0, 2, (population_size, size * size))
    t_max = FES // population_size

    quality = quality_func(population, size)[0]
    best_solution = population[np.argmax(quality)]
    best_score = np.max(quality)

    timer_start = time.time()

    for i in range(t_max):
        # after every percent of t_max pront current best score
        if i % (t_max // 100) == 0:
            print(f"Best score after {i / t_max * 100:.2f}%: {best_score}")

        new_population = roulette_selection(population, quality, population_size)
        new_population = crossover(new_population, crossover_prob)
        new_population = mutation(new_population, mutation_prob)

        quality = quality_func(new_population, size)[0]
        best_id = np.argmax(quality)
        best_new_solution = new_population[best_id]
        best_new_score = quality[best_id]
        if best_new_score > best_score:
            best_solution = best_new_solution
            best_score = best_new_score

        new_population[0] = best_solution
        population = new_population

    timer_end = time.time()
    elapsed_time = timer_end - timer_start
    print(f"Elapsed time: {elapsed_time:.2f}s")

    return best_solution, best_score


def trivial_solution(size):
    vector = []
    for i in range(size * size):
        if i % 2 == 0:
            vector.append(1)
        else:
            vector.append(0)
    vector = np.array(vector)
    score1, grid1, points1 = evaluate(vector, size)
    plots.visualize(grid1, points1, score1, 20)
    print(f"Wynik rozwiązania trywialnego: {score1}")


def smart_solution(size):
    vector = []
    for i in range(size):
        for j in range(size):
            if (i % 4 == 0 and j % 2 == 0) or (i % 4 == 2 and j % 2 == 1):
                vector.append(1)
            else:
                vector.append(0)
    vector = np.array(vector)
    score1, grid1, points1 = evaluate(vector, size)
    plots.visualize(grid1, points1, score1, 20)
    print(f"Wynik rozwiązania trywialnego: {score1}")


def random_solution(size, num_points):
    vectors = []
    for i in range(num_points):
        vector = np.random.randint(0, 2, size * size)
        vectors.append(vector)
    scores, grids, points = evaluate(vectors, size)
    print(f"Średni wynik rozwiązania losowego: {np.mean(scores)}")


def test_pop_size_impact(population_sizes=None, mutation_prob=0.05, crossover_prob=0.8, FES=4_000_000, size=20,
                         filename='plik.txt'):
    results = []
    with open(filename, 'w') as file:
        for population_size in population_sizes:
            best_solution, best_score = evolutionary_algorithm(
                population_size, mutation_prob, crossover_prob, FES, size
            )
            file.write(f"Population size: {population_size}\n")
            file.write(f"Best score: {best_score}\n")
            file.write(f"Best solution: {best_solution}\n")
            results.append(best_score)
            print(f"Population size {population_size}: Best score = {best_score}")
    return results


def test_mut_prob_impact(population_size=300, mutation_probs=None, crossover_prob=0.8, FES=4_000_000, size=20,
                         filename='plik.txt'):
    results = []
    with open(filename, 'w') as file:
        for mutation_prob in mutation_probs:
            best_solution, best_score = evolutionary_algorithm(
                population_size, mutation_prob, crossover_prob, FES, size
            )
            file.write(f"Mutation prob: {mutation_prob}\n")
            file.write(f"Best score: {best_score}\n")
            file.write(f"Best solution: {best_solution}\n")
            results.append(best_score)
            print(f"Mutation prob {mutation_prob}: Best score = {best_score}")
    return results


def test_recom_prob_impact(population_size=300, mutation_prob=0.05, crossover_probs=None, FES=4_000_000, size=20,
                           filename='plik.txt'):
    results = []
    with open(filename, 'w') as file:
        for crossover_prob in crossover_probs:
            best_solution, best_score = evolutionary_algorithm(
                population_size, mutation_prob, crossover_prob, FES, size
            )
            file.write(f"Crossover prob: {crossover_prob}\n")
            file.write(f"Best score: {best_score}\n")
            file.write(f"Best solution: {best_solution}\n")
            results.append(best_score)
            print(f"Crossover prob {crossover_prob}: Best score = {best_score}")
    return results


# trivial_solution(20)
# smart_solution(20)
# random_solution(20, 100)

# Hierparametry dające wyniki lepsze niż losowe
population_size = 300
mutation_prob = 0.05
recombination_prob = 0.8
FES = 4_000_000
size = 20

# evolutionary_algorithm(population_size, mutation_prob, recombination_prob, FES, size)
# Uzyskany wynik 252
#
# # Wywołanie testów z różnymi wielkościami populacji
# population_sizes = [2, 4, 6, 8, 10, 12, 20, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000, 8000, 10000]
# test_pop_size_impact(population_sizes=population_sizes, filename="1.txt")
#
# mutation_probs = [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1]
# test_mut_prob_impact(mutation_probs=mutation_probs, filename="2.txt")
#
# recom_probs = [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1]
# test_recom_prob_impact(crossover_probs=recom_probs, filename="3.txt")

# plots.plot_data_from_file("different_population_sizes.txt", "population")
# plots.plot_data_from_file("different_mutation_probs.txt", "mutation")
# plots.plot_data_from_file("different_recombination_probs.txt", "crossover")


population_sizes = [2, 4, 6]
mutation_probs = [0.001, 0.003, 0.005]
recombination_probs = [0.1, 0.2, 0.3]

best_score = 0
best_solution = None

for population_size in population_sizes:
    for mutation_prob in mutation_probs:
        for recombination_prob in recombination_probs:
            solution, score = evolutionary_algorithm(
                population_size, mutation_prob, recombination_prob, FES, size
            )
            if score > best_score:
                best_score = score
                best_solution = solution
            print(f"Population size: {population_size}; Mutation prob: {mutation_prob}; Crossover prob: {recombination_prob}; Best score: {score}")




score, grid, points = evaluate(best_solution, 20)
plots.visualize(grid, points, score, 20)

