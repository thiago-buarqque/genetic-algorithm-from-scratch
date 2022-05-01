import math
import numpy as np
from builder import Builder

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

# def fitness_func(genes):
#     x1 = genes[0]
#     x2 = genes[1]
#     x3 = genes[2]
#     x4 = genes[3]
#     x5 = genes[4]

#     return (1 - (abs((math.sqrt(x1) * math.sin(x2**(math.sqrt(x3)))))) - ((x5**2)/x4) + (1/math.sqrt(x5))).real


# def plot_fitness_func(x1, x2, x3, x4, x5):
#     return (1 - (abs((sqrt(x1) * sin(x2**(sqrt(x3)))))) - ((x5**2)/x4) + (1/sqrt(x5))).real

## Cool functions
## 1) 1 - (abs((math.sqrt(x1) * math.sin(x2**(math.sqrt(x3)))))) - ((x5**2)/x4) + (1/math.sqrt(x5))
## 2) ((x1 + 2*x2 - 7)**2) + ((2*x1 + x2 - 5)**2)
## 3) Eggholder: -((x2 + 47) * sin(sqrt(abs((x1 / 2) + (x2 + 47))))) - (x1 * sin(sqrt(abs(x1 - (x2 + 47)))))

user_func = None


def fitness_func(func_vars):
    # Func vars had a type of {"var_name": var, ...}
    func_vars['np'] = np
    func_vars['math'] = math

    return eval(user_func, func_vars)


def plot_fitness_func(x1, x2):
    # Func vars had a type of {"var_name": var, ...}
    func_vars = {'x1': x1, 'x2': x2}
    func_vars['np'] = np
    func_vars['math'] = math

    return eval(user_func, func_vars)


def mutation_func(ind):
    gene_i = np.random.randint(low=0, high=len(ind.genes))

    lower_bound = ind.complex_genes[gene_i]['lower_bound']
    upper_bound = ind.complex_genes[gene_i]['upper_bound']

    ind.genes[gene_i] = ind.complex_genes[gene_i]["gen_func"](
        low=lower_bound, high=upper_bound)


if __name__ == '__main__':
    builder = Builder()

    user_func = builder.construct_fitness_function()
    base_ind = builder.construct_base_individual()
    toolbox = builder.construct_toolbox(
        fitness_func, mutation_func, plot_fitness_func)
    ga = builder.construct_ga_instance(base_ind, toolbox)

    ga.optimize()
