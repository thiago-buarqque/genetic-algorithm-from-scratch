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

        convergence_data = [None]*generations

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

            length = len(self.pop)
            mean = sum(fits) / length

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

            convergence_data[i] = best_ind.fitness

            print(f"Gen (%2d) - Min: %5.4f - Avg: %5.4f - Gen max: %5.4f -"
                  f" Gen best ind: {gen_best_ind} - Best ind (fitness): {best_ind} (%5.4f)" %
                  (i+1, min(fits), mean, max(fits), best_ind.fitness))

            if early_stop_counter == early_stop_strg[0]:
                print(f'Early stopping!')
                break

        print(
            f'The best one is this guy: {best_ind.genes}, with the fitness of {best_ind.fitness}')

        self.plot_convergence_curve(convergence_data)
        self.plot_objective_function(best_ind=best_ind)

    def plot_convergence_curve(self, convergence_data):
        # Data for plotting
        t = np.arange(len(convergence_data))
        s = convergence_data

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='Generation', ylabel='Fitness (max)',
            title='Convergence curve')
        ax.grid()

        fig.savefig("convergence_curve.png")
        # plt.show()

    def plot_objective_function(self, best_ind):
        if len(best_ind.genes) > 2:
            return

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
        # plt.show()
