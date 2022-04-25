import random


class CrossoverToolbox:
    def cs_uniform(self, ind1, ind2):
        for i in range(len(ind1.genes)):
            if random.random() < 0.5:
                temp = ind1.genes[i]
                ind1.genes[i] = ind2.genes[i]
                ind2.genes[i] = temp
        return ind1, ind2
