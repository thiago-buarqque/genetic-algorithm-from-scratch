class GA:
    def __init__(self, crossover_func, crossover_rate, generations, population_size, mutation_func, mutation_rate, use_elitism=False, elitism_size=None):
        self.crossover_func = crossover_func
        self.crossover_rate = crossover_rate

        self.max_generations = generations
        self.generation_size = population_size

        self.mutation_func = mutation_func
        self.mutation_rate = mutation_rate

        self.use_elitism = use_elitism
        self.elitism_size = elitism_size
        
        if use_elitism is True and (elitism_size is None or elitism_size == 0):
            raise Exception("Elitism size is invalid")
            
