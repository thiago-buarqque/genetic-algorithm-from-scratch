from cmath import sin, sqrt
import math
import numpy as np
from builder import Builder

from ga.Individual import Individual

import matplotlib.pyplot as plt
from ga.operations import Operations

from ga.toolbox import Toolbox
from ga.ga import GA


# def fitness_func(genes):
#     x1 = genes[0]
#     x2 = genes[1]

#     return (-((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))).real


# def plot_fitness_func(x1, x2):
#     return (-((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))).real

# def fitness_func(genes):
#     x1 = genes[0]
#     x2 = genes[1]

#     return (((x1 + 2*x2 - 7)**2) + ((2*x1 + x2 - 5)**2)).real


# def plot_fitness_func(x1, x2):
#     return (((x1 + 2*x2 - 7)**2) + ((2*x1 + x2 - 5)**2)).real

def fitness_func(genes):
    x1 = genes[0]
    x2 = genes[1]
    x3 = genes[2]
    x4 = genes[3]
    x5 = genes[4]
           
    return (1 - (abs((sqrt(x1) * sin(x2**(sqrt(x3)))))) - ((x5**2)/x4) + (1/sqrt(x5))).real


def plot_fitness_func(x1, x2, x3, x4, x5):
    return (1 - (abs((sqrt(x1) * sin(x2**(sqrt(x3)))))) - ((x5**2)/x4) + (1/sqrt(x5))).real

def mutation_func(ind):
    gene_i = np.random.randint(low=0, high=len(ind.genes))

    lower_bound = ind.complex_genes[gene_i]['lower_bound']
    upper_bound = ind.complex_genes[gene_i]['upper_bound']

    # Allow the user to choose np.random.func_type
    ind.genes[gene_i] = np.random.randint(low=lower_bound, high=upper_bound)


if __name__ == '__main__':
    operations = Operations()
    toolbox = Toolbox()

    base_ind = Individual()
    base_ind.register("x1", np.random.randint, 0, 2048)
    base_ind.register("x2", np.random.randint, 0, 512)
    base_ind.register("x3", np.random.randint, 1, 2048)
    base_ind.register("x4", np.random.randint, 1, 120)
    base_ind.register("x5", np.random.randint, 1, 10000)

    toolbox.register("select", operations.sl_tournament)
    toolbox.register("crossover", operations.cs_uniform)
    toolbox.register("mutate", mutation_func)
    toolbox.register("evaluate", fitness_func)

    # Temporally, I need to figure out how to pass the X and Y from plot
    toolbox.register("plot_eval_func", plot_fitness_func)

    ga = GA(
        base_ind=base_ind,
        crossover_rate=0.9,
        mutation_rate=0.05,
        population_size=100,
        toolbox=toolbox
    )

    ga.optimize(
        generations=50,
        early_stop_strg=[None, None]
    )
