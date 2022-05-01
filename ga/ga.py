from cmath import sin, sqrt
from operator import attrgetter
from tkinter import E
import numpy as np

import matplotlib.pyplot as plt


def fitness_func_(x1, x2):
    return (-((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))).real


class GA:
    def __init__(self,
                 crossover_rate,
                 population_size,
                 mutation_rate,
                 toolbox,
                 base_ind,
                 use_elitism=False,
                 elitism_size=None):
        self.crossover_rate = crossover_rate

        self.population_size = population_size

        self.mutation_rate = mutation_rate

        self.elitism_size = elitism_size
        self.use_elitism = use_elitism

        if use_elitism is True and (elitism_size is None or elitism_size == 0):
            raise Exception("Elitism size is invalid")

        self.base_ind = base_ind

        self.pop = []
        self.toolbox = toolbox

        self.convergence_data = []

        self.gen_min_data = []
        self.gen_avg_data = []
        self.gen_max_data = []

        self.set_up()

    def set_up(self):
        self.pop = self.toolbox.generate_population(
            self.base_ind, self.population_size, self.toolbox.evaluate)
        self.evaluate_population()

    def evaluate_population(self):
        for ind in self.pop:
            ind.evaluate()

    def optimize(self, generations, early_stop_strg=None):
        best_ind = None

        self.convergence_data = []

        early_stop_last_ind = None
        early_stop_counter = 0
        for i in range(generations):
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

            # Early stopping conditions
            gen_best_ind = max(self.pop, key=attrgetter('fitness'))
            if best_ind == None or gen_best_ind.fitness > best_ind.fitness:
                if early_stop_strg[1] is not None:
                    if early_stop_strg[1] < (gen_best_ind.fitness - early_stop_last_ind.fitness):
                        early_stop_last_ind = gen_best_ind
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1
                else:
                    early_stop_counter = 0

                best_ind = gen_best_ind
            else:
                early_stop_counter += 1

            self.convergence_data.append(best_ind.fitness)

            length = len(self.pop)
            mean = sum(fits) / length
            _min = min(fits)
            _max = max(fits)

            print(f"Gen (%2d) - Min: %5.4f - Avg: %5.4f - Gen max: %5.4f -"
                  f" Gen best ind: {gen_best_ind} - Best ind (fitness): {best_ind} (%5.4f)" %
                  (i+1, _min, mean, _max, best_ind.fitness))

            self.gen_min_data.append(_min)
            self.gen_avg_data.append(mean)
            self.gen_max_data.append(_max)

            if early_stop_counter == early_stop_strg[0]:
                print(f'Early stopping at gen {i+1}!')
                break

        self.plot_convergence_curve()
        self.plot_min_avg_max()
        self.plot_objective_function(best_ind=best_ind)

    def plot_convergence_curve(self):
        t = np.arange(len(self.convergence_data))
        s = self.convergence_data

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='Generation', ylabel='Fitness (max)',
            title='Convergence curve')
        ax.grid()

        fig.savefig("convergence_curve.png")
        # plt.show()

    def plot_min_avg_max(self):
        gens = np.arange(len(self.gen_min_data))
        min_data = self.gen_min_data
        avg_data = self.gen_avg_data
        max_data = self.gen_max_data

        fig, ax = plt.subplots()
        plt.plot(gens, min_data, label="min")
        plt.plot(gens, avg_data, label="avg")
        plt.plot(gens, max_data, label="max")

        plt.legend()

        ax.set(xlabel='Generation', ylabel='Fitness',
            title='Min, avg, max fitness p/ gen.')
        ax.grid()

        fig.savefig("min_avg_max_curves.png")
        plt.show()

    def plot_objective_function(self, best_ind):
        if len(best_ind.genes) > 2:
            return

        if len(best_ind.genes) == 2:
            x = np.linspace(
                best_ind.complex_genes[0]["lower_bound"], best_ind.complex_genes[0]["upper_bound"], 50)
            y = np.linspace(
                best_ind.complex_genes[1]["lower_bound"], best_ind.complex_genes[1]["upper_bound"], 50)

            X, Y = np.meshgrid(x, y)
            Z = np.vectorize(self.toolbox.plot_eval_func)(X, Y)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')
            ax.contour3D(X, Y, Z, 50, cmap='binary')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            # Plot best individual position
            ax.scatter([best_ind.genes[0]], [best_ind.genes[1]],
                    [best_ind.fitness], plotnonfinite=True, zorder=10)
            
            fig.colorbar(surf, shrink=0.5, aspect=5)
            fig.savefig("best_individual_3d.png")
            plt.show()