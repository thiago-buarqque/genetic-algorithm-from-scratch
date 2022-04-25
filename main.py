import numpy as np

from genetic_operators.crossover.crossoverToolbox import CrossoverToolbox
from genetic_operators.selection.selectionToolbox import SelectionToolbox
from individual.Individual import Individual


def fitness_func(*args):
    return np.random.uniform(low=0, high=1)

def mutation_func(ind):
    min_bound = -100
    max_bound = 100

    gene_i = np.random.randint(low=0, high=len(ind.genes), size=1)
    ind.genes[gene_i] = np.random.randint(low=min_bound, high=max_bound+1, size=1)

if __name__ == '__main__':
    pop = []
    for i in range(10):
        ind_genes = np.random.randint(low=-100, high=101, size=4)
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
    new_pop = sl_toolbox.sl_tournament(pop)

    new_pop_str = []
    for ind in new_pop:
        new_pop_str.append(ind.__str__())

    # print(f'max: {max(pop_str)} min: {min(pop_str)} - {pop_str}')
    # print(f'max: {max(new_pop_str)} min: {min(new_pop_str)} - {new_pop_str}')

    print(f'Old pop: {pop_str}')
    print(f'New pop: {new_pop_str}\n')

    for child1, child2 in zip(new_pop[::2], new_pop[1::2]):
        if np.random.uniform() < 0.75:
            cs_toolbox.cs_uniform(child1, child2)
            child1.fitness = None
            child2.fitness = None

    new_pop_str = []
    for ind in new_pop:
        new_pop_str.append(ind.__str__())

    print(f'Offspring')
    print(new_pop_str)

    for mutant in new_pop:
        if np.random.uniform() < 0.15:
            mutation_func(mutant)
            mutant.fitness = None

    new_pop_str = []
    for ind in new_pop:
        new_pop_str.append(ind.__str__())

    print(f'Final pop')
    print(new_pop_str)