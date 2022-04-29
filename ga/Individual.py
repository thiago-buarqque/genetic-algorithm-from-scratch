class Individual:
    def __init__(self, genes, fitness_function):
        self.genes = genes
        self.fitness = 0

        self.fitness_function = fitness_function

    def evaluate_fitness(self):
        self.fitness = self.fitness_function(self.genes)

    def __str__(self):
        return f"{self.genes}"
