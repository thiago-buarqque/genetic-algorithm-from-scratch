import copy
from operator import attrgetter
import random

import numpy as np


class ToolBox:
    def sl_tournament(self, pop, tourn_size=2):
        new_population = []
        for i in range(len(pop)):
            selected_individuals = np.random.choice(
                pop, size=tourn_size, replace=False)

            winner = max(selected_individuals,
                         key=lambda ind: ind.fitness if ind.fitness > ind.fitness else ind.fitness)
            new_population.append(copy.deepcopy(winner))

        return new_population

    def sl_roulette_wheel(self, pop):
        s_inds = sorted(pop, key=attrgetter("fitness"), reverse=True)
        sum_fits = sum(getattr(ind, "fitness") for ind in pop)
        chosen = []
        for i in range(len(pop)):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += getattr(ind, "fitness")
                if sum_ > u:
                    chosen.append(ind)
                    break

        return chosen


    def cs_uniform(self, ind1, ind2):
        for i in range(len(ind1.genes)):
            if random.random() < 0.5:
                temp = ind1.genes[i]
                ind1.genes[i] = ind2.genes[i]
                ind2.genes[i] = temp
        return ind1, ind2
