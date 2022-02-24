from src.benchmark import *
pi = 3.141592653589793

general_functions = [
    dict(name="func1", function=func1, boundary=(-20, 20), num_particles=200, epochs=200, best_fitness=-10),
    dict(name="func2", function=func2, boundary=(-10, 10), num_particles=200, epochs=200, best_fitness=0),
    dict(name="func3", function=func3, boundary=(-20, 20), num_particles=200, epochs=200, best_fitness=4),
    dict(name="func4", function=func4, boundary=(-20, 20), num_particles=200, epochs=200, best_fitness=4),
    dict(name="func5", function=func5, boundary=(-5, 5), num_particles=200, epochs=200, best_fitness=-2),
]

two_dims_functions = [
    dict(name="drop_wave", function=drop_wave, boundary=(-5.12, 5.12), num_particles=100, epochs=200, best_fitness=-1),
    dict(name="cross_in_tray", function=cross_in_tray, boundary=(-10, 10), num_particles=100, epochs=200, best_fitness=-2.06261),
    dict(name="bohachevsky", function=bohachevsky, boundary=(-100, 100), num_particles=100, epochs=200, best_fitness=0),
    dict(name="sphere", function=sphere, boundary=(-5.12, 5.12), num_particles=100, epochs=200, best_fitness=0),
    dict(name="booth", function=booth, boundary=(-10, 10), num_particles=100, epochs=200, best_fitness=0),
    dict(name="zakharov", function=zakharov, boundary=(-5, 10), num_particles=100, epochs=200, best_fitness=0),
    dict(name="three_hump_camel", function=three_hump_camel, boundary=(-5, 5), num_particles=100, epochs=200, best_fitness=0),
    dict(name="dixon_prince", function=dixon_prince, boundary=(-10, 10), num_particles=100, epochs=200, best_fitness=0),
    dict(name="easom", function=easom, boundary=(-100, 100), num_particles=100, epochs=200, best_fitness=-1),
    dict(name="michalewicz", function=michalewicz, boundary=(0, pi), num_particles=100, epochs=200, best_fitness=-1.8013),
    dict(name="beale", function=beale, boundary=(-4.5, 4.5), num_particles=100, epochs=200, best_fitness=0),
    dict(name="goldstein_price", function=goldstein_price, boundary=(-2, 2), num_particles=100, epochs=200, best_fitness=3),
]


d_dims_functions = [
    dict(name="sphere", function=sphere, boundary=(-5.12, 5.12), num_particles=100, epochs=200, best_fitness=0),
    dict(name="zakharov", function=zakharov, boundary=(-5, 10), num_particles=100, epochs=200, best_fitness=0),
    dict(name="dixon_prince", function=dixon_prince, boundary=(-10, 10), num_particles=100, epochs=200, best_fitness=0),
]