class Builder:
    def __init__(self):
        self.user_func = None

    def construct_fitness_function(self):
        in_func = input("Enter your function. You can use math. Python math and numpy (np)")
        in_func = compile(in_func)

        return in_func