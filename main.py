from cmath import sin, sqrt
import numpy as np

from ga.Individual import Individual

import matplotlib.pyplot as plt
from ga.operations import Operations

from ga.toolbox import Toolbox
from ga.ga import GA


def fitness_func(genes):
    x1 = genes[0]
    x2 = genes[1]

    return (-((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))).real


def fitness_func_(x1, x2):
    return (-((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))).real


def mutation_func(ind):
    gene_i = np.random.randint(low=0, high=len(ind.genes))

    lower_bound = ind.complex_genes[gene_i]['lower_bound']
    upper_bound = ind.complex_genes[gene_i]['upper_bound']

    # Make np.random.func_type be chosed by the user
    ind.genes[gene_i] = np.random.randint(
        low=lower_bound, high=upper_bound + 1)


if __name__ == '__main__':
    operations = Operations()
    toolbox = Toolbox()

    base_ind = Individual()
    base_ind.register("x1", np.random.randint, -512, 512)
    base_ind.register("x2", np.random.randint, -512, 512)

    toolbox.register("select", operations.sl_tournament)
    toolbox.register("crossover", operations.cs_uniform)
    toolbox.register("mutate", mutation_func)
    toolbox.register("evaluate", fitness_func)

    ga = GA(
        base_ind=base_ind,
        crossover_rate=0.75,
        generations=100,
        mutation_rate=0.2,
        population_size=100,
        toolbox=toolbox
    )

    ga.optimize()

    # print(f'{toolbox.evaluate(43, 29)}')
    # print(f'A new pop:')
    # print(toolbox.generate_population(base_ind, 5)[0])
    # ind1 = Individual(genes=[900, 123], fitness_function=fit)
    # ind2 = Individual(genes=[342, 53], fitness_function=fit)
    # ind3 = Individual(genes=[101, -23], fitness_function=fit)
    # ind4 = Individual(genes=[10, -123], fitness_function=fit)
    # ind5 = Individual(genes=[-621, 144], fitness_function=fit)

    # ind1.evaluate_fitness()
    # ind2.evaluate_fitness()
    # ind3.evaluate_fitness()
    # ind4.evaluate_fitness()
    # ind5.evaluate_fitness()

    # print(
    #     f'Fitnesses iniciais: {ind1.fitness},{ind2.fitness}, {ind3.fitness}, {ind4.fitness}, {ind5.fitness}')

    # pop = [ind1, ind2, ind3, ind4, ind5]

    # toolbox = ToolBox()

    # print()
    # for ind in toolbox.sl_roulette_wheel(pop):
    #     print(ind.fitness)

    # ga = GA(crossover_func=1, crossover_rate=0.9, generations=100,
    #         population_size=100, mutation_func=1, mutation_rate=0.05, use_elitism=True)
    # min_bound = -512
    # max_bound = 512
    # pop = []

    # x = np.linspace(min_bound, max_bound + 1, 30)
    # y = np.linspace(min_bound, max_bound + 1, 30)

    # X, Y = np.meshgrid(x, y)
    # Z = np.vectorize(fitness_func_)(X, Y)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    # #                 cmap='viridis', edgecolor='none')
    # ax.contour3D(X, Y, Z, 50, cmap='binary')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # for i in range(50):
    #     ind_genes = np.random.uniform(
    #         low=min_bound, high=max_bound + 1, size=2)
    #     pop.append(Individual(
    #         genes=ind_genes,
    #         fitness_function=fitness_func
    #     ))

    #     pop[i].evaluate_fitness()

    # pop_str = []
    # for ind in pop:
    #     pop_str.append(ind.__str__())

    # toolbox = ToolBox()

    # bests_x = []
    # bests_y = []
    # bests_z = []

    # best_ind = []
    # for i in range(50):
    #     offspring = toolbox.sl_tournament(pop)

    #     # Perform crossover
    #     for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #         if np.random.uniform() < 0.75:
    #             toolbox.cs_uniform(child1, child2)
    #             child1.fitness = None
    #             child2.fitness = None

    #     # Perform mutation
    #     for ind in [ind for ind in offspring if ind.fitness is None]:
    #         if np.random.uniform() < 0.15:
    #             mutation_func(ind, min_bound, max_bound)
    #             ind.fitness = None

    #     for invalid_ind in [ind for ind in offspring if ind.fitness is None]:
    #         invalid_ind.evaluate_fitness()

    #     pop = offspring

    #     fits = [ind.fitness for ind in pop]

    #     length = len(pop)
    #     mean = sum(fits) / length
    #     sum2 = sum(x * x for x in fits)

    #     best_ind = max(pop, key=attrgetter('fitness'))
    #     print(
    #         f"Min: {min(fits)} - Avg: {mean} - Max: {max(fits)} - Best: {best_ind}")

    #     # ax.scatter(best_ind.genes[0], best_ind.genes[1], fitness_func(best_ind.genes))
    #     bests_x.append(best_ind.genes[0])
    #     bests_y.append(best_ind.genes[1])
    #     bests_z.append(max(fits))

    # ax.scatter(bests_x, bests_y, bests_y)
    # plt.show()
