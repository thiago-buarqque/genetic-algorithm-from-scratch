import numpy as np

from ga.Individual import Individual
from ga.ga import GA
from ga.operations import Operations
from ga.toolbox import Toolbox


class Builder:
    def construct_ga_instance(self, base_ind, toolbox):
        print("\nNow let's get the parameters.\n")

        crossover_rate = float(input("Crossover rate: "))
        mutation_rate = float(input("Mutation rate: "))
        population_size = int(input("Population size: "))
        max_generations = int(input("Generations: "))
        elite_size = int(input("Elite size (0 if you doesn't want it): "))

        use_early_stop = input("Do you want to use early stopping (y/n): ")

        early_stop_size = None
        if use_early_stop == 'y':
            early_stop_size = int(input("How many generation: "))

        return GA(
            base_ind=base_ind,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            population_size=population_size,
            toolbox=toolbox,
            max_generations=max_generations,
            elite_size=elite_size,
            early_stop_strg=[early_stop_size, None]
        )

    def construct_toolbox(self, fitness_func, mutation_func, plot_fitness_func):
        operations = Operations()
        toolbox = Toolbox()

        print("\nNow let's choose the genetic operations.\n")

        print("Which selection strategy do you want to use? (default: 2): ")
        print("1- Roulette wheel")
        print("2- Tournament of 'n' individuals")
        selection_strg = input(": ")

        if selection_strg == '2':
            tourn_size = int(input("\nHow many individual in each tournament? (default: 2): "))
            toolbox.register("select", operations.sl_tournament, tourn_size=tourn_size)
        else:
            toolbox.register("select", operations.sl_roulette_wheel)

        print("\n\nWhich crossover strategy do you want to use? (default: 1): ")
        print("1- One point")
        print("2- Two points")
        print("3- Uniform")
        crossover_strg = input(": ")

        crossover_func = operations.cs_one_point
        if crossover_strg == '2':
            crossover_func = operations.cs_two_point
        elif crossover_strg == '3':
            crossover_func = operations.cs_uniform
        
        toolbox.register("crossover", crossover_func)
        toolbox.register("evaluate", fitness_func)
        toolbox.register("mutate", mutation_func)
        toolbox.register("plot_eval_func", plot_fitness_func)

        return toolbox

    def construct_base_individual(self):
        base_ind = Individual()

        while(True):
            print("\nType 'exit' any time when you finish adding all function's variables\n")
            var_name = input("Type the var name: ")
            if var_name == 'exit':
                break

            var_lower_bound = input("Type the var lower bound: ")
            if var_lower_bound == 'exit':
                break

            var_upper_bound = input("Type the var upper bound: ")
            if var_upper_bound == 'exit':
                break

            var_type = input("Is this var decimal or integer?")
            if var_type == 'exit':
                break

            gen_func = None
            if var_type == 'integer':
                gen_func = np.random.randint
                var_lower_bound = int(var_lower_bound)
                var_upper_bound = int(var_upper_bound)
            else:
                gen_func = np.random.uniform
                var_lower_bound = float(var_lower_bound)
                var_upper_bound = float(var_upper_bound)

            base_ind.register(var_name, gen_func,
                              var_lower_bound, var_upper_bound)

        return base_ind

    def construct_fitness_function(self):
        in_func = input(
            "Enter your function. You can use Python python built-in functions, math and numpy (np): ")
        in_func = compile(in_func, 'input', 'eval')

        return in_func
