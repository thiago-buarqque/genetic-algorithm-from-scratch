import numpy as np


class SelectionToolbox:
    def sl_tournament(self, pop, tourn_size=2):
        new_population = []
        for i in range(len(pop)):
            selected_individuals = np.random.choice(pop, size=tourn_size, replace=False)

            winner = max(selected_individuals, key=lambda ind: ind.fitness if ind.fitness > ind.fitness else ind.fitness)
            new_population.append(winner)

        return new_population
