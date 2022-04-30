from cmath import sin, sqrt
from operator import attrgetter
import random
import numpy as np

from ga.Individual import Individual

import matplotlib.pyplot as plt


def fitness_func_(x1, x2):
    return (-((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))).real


class GA:
    def __init__(self,
                 crossover_rate,
                 generations,
                 population_size,
                 mutation_rate,
                 toolbox,
                 base_ind,
                 use_elitism=False,
                 elitism_size=None):
        self.crossover_rate = crossover_rate

        self.max_generations = generations
        self.population_size = population_size

        self.mutation_rate = mutation_rate

        self.elitism_size = elitism_size
        self.use_elitism = use_elitism

        if use_elitism is True and (elitism_size is None or elitism_size == 0):
            raise Exception("Elitism size is invalid")

        self.base_ind = base_ind

        self.pop = []
        self.toolbox = toolbox

        self.set_up()

    def set_up(self):
        self.pop = self.toolbox.generate_population(
            self.base_ind, self.population_size, self.toolbox.evaluate)
        self.evaluate_population()

    def evaluate_population(self):
        for ind in self.pop:
            ind.evaluate_function()

    # def plot_fitness_func_based_on_range(self):
        # x = np.linspace(min_bound, max_bound + 1, 30)
        # y = np.linspace(min_bound, max_bound + 1, 30)

        # X, Y = np.meshgrid(x, y)
        # Z = np.vectorize(self.fitness_func)(X, Y)

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
        #                 cmap='viridis', edgecolor='none')
        # ax.contour3D(X, Y, Z, 50, cmap='binary')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

    def evaluate_population(self):
        for ind in self.pop:
            ind.evaluate()

    def optimize(self):
        best_ind = None

        # self.evaluate_population()

        for i in range(self.max_generations):
            offspring = self.toolbox.select(self.pop)

            # Perform crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.uniform() < self.crossover_rate:
                    self.toolbox.crossover(child1, child2)
                    child1.fitness = None
                    child2.fitness = None

            # Perform mutation
            for ind in [ind for ind in offspring if ind.fitness is None]:
                if np.random.uniform() < self.mutation_rate:
                    self.toolbox.mutate(ind)
                    ind.fitness = None

            # Evaluate invalid individuals
            for invalid_ind in [ind for ind in offspring if ind.fitness is None]:
                invalid_ind.evaluate()

            self.pop = offspring

            fits = [ind.fitness for ind in self.pop]

            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)

            gen_best_ind = max(self.pop, key=attrgetter('fitness'))
            if best_ind == None or gen_best_ind.fitness > best_ind.fitness:
                best_ind = gen_best_ind

            print(
                f"Gen ({i+1}) - Min: {min(fits)} - Avg: {mean} - Max: {max(fits)} - Best ind. generation: {best_ind}")

        x = np.linspace(-512, 512 + 1, 100)
        y = np.linspace(-512, 512 + 1, 100)

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

        best_x, best_y, best_z = [], [], []

        best_x.append(best_ind.genes[0])
        best_y.append(best_ind.genes[1])
        best_z.append(best_ind.fitness)

        ax.scatter(best_x, best_y, best_y)
        plt.show()
