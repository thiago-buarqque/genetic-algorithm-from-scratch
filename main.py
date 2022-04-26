from cmath import sin, sqrt
import numpy as np

from operator import attrgetter

from genetic_operators.crossover.crossoverToolbox import CrossoverToolbox
from genetic_operators.selection.selectionToolbox import SelectionToolbox
from individual.Individual import Individual

import matplotlib.pyplot as plt


def fitness_func(genes):
    x1 = genes[0]
    x2 = genes[1]

    return -((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))


def fitness_func_(x1, x2):
    return -((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))


def mutation_func(ind, min_bound, max_bound):
    gene_i = np.random.randint(low=0, high=len(ind.genes), size=1)
    ind.genes[gene_i] = np.random.uniform(
        low=min_bound, high=max_bound + 1, size=1)


if __name__ == '__main__':
    min_bound = -512
    max_bound = 512
    pop = []

    x = np.linspace(min_bound, max_bound + 1, 30)
    y = np.linspace(min_bound, max_bound + 1, 30)

    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(fitness_func_)(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                 cmap='viridis', edgecolor='none')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i in range(50):
        ind_genes = np.random.uniform(low=min_bound, high=max_bound + 1, size=2)
        pop.append(Individual(
            genes=ind_genes,
            fitness_function=fitness_func
        ))

        pop[i].evaluate_fitness()

    pop_str = []
    for ind in pop:
        pop_str.append(ind.__str__())

    sl_toolbox = SelectionToolbox()
    cs_toolbox = CrossoverToolbox()

    bests_x = []
    bests_y = []
    bests_z = []

    best_ind = []
    for i in range(50):
        offspring = sl_toolbox.sl_tournament(pop)

        # Perform crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.uniform() < 0.75:
                cs_toolbox.cs_uniform(child1, child2)
                child1.fitness = None
                child2.fitness = None

        # Perform mutation
        for ind in [ind for ind in offspring if ind.fitness is None]:
            if np.random.uniform() < 0.15:
                mutation_func(ind, min_bound, max_bound)
                ind.fitness = None

        for invalid_ind in [ind for ind in offspring if ind.fitness is None]:
            invalid_ind.evaluate_fitness()

        pop = offspring

        fits = [ind.fitness for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)

        best_ind = max(pop, key=attrgetter('fitness'))
        print(f"Min: {min(fits)} - Avg: {mean} - Max: {max(fits)} - Best: {best_ind}")

        # ax.scatter(best_ind.genes[0], best_ind.genes[1], fitness_func(best_ind.genes))
        bests_x.append(best_ind.genes[0])
        bests_y.append(best_ind.genes[1])
        bests_z.append(max(fits))

    ax.scatter(bests_x, bests_y, bests_y)
    plt.show()
