import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from pyswarms.single import GlobalBestPSO


def genetic_algorithm(
    fitness,
    dims,
    bounds,
    max_iter,
    num_pop=100,
):

    varbound = np.array([list(bounds)] * dims)

    model = ga(
        function=fitness,
        dimension=dims,
        variable_type="real",
        variable_boundaries=varbound,
        algorithm_parameters={
            "max_num_iteration": max_iter,
            "population_size": num_pop,
            "mutation_probability": 0.1,
            "elit_ratio": 0.01,
            "crossover_probability": 0.5,
            "parents_portion": 0.3,
            "crossover_type": "uniform",
            "max_iteration_without_improv": None,
        },
        convergence_curve=False,
        progress_bar=False,
    )

    model.run()

    return model.report[1:], model.best_variable.tolist()


def differential_evolution(
    fobj,
    dims,
    bounds,
    mut=0.3,
    crossp=0.2,
    popsize=100,
    its=1000,
):
    dimensions = dims
    bounds = [bounds] * dims
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    history = []

    for _ in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        history.append(fitness[best_idx])

    return history, list(best)


def pso(
    fitness,
    dims,
    bounds,
    max_iter,
    num_pop=100,
):

    bound_min = np.array([bounds[0]] * dims)
    bound_max = np.array([bounds[1]] * dims)

    pso = GlobalBestPSO(
        n_particles=num_pop,
        dimensions=dims,
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        bounds=(bound_min, bound_max),
    )

    _, pos = pso.optimize(fitness, iters=max_iter, verbose=False)

    return pso.cost_history, pos.tolist()


def simulated_annealing(objective, dims, bounds, max_iter, step_size=0.1, temp=10):
    # generate an initial point
    if not isinstance(bounds, np.ndarray):
        bounds = np.array([bounds])
    bounds = np.repeat(bounds, dims, axis=0)
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    history = [best_eval]
    # run the algorithm
    for i in range(max_iter):
        # take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            # print('>%d f(%s) = %.5f' % (i, best, best_eval))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = np.exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or np.random.rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval

        history.append(best_eval)
    return (history[1:], best.tolist())


# from src.fitness import sphere
# from src.pso import PSOL
# from pprint import pprint


# res1 = differential_evolution(sphere, 2, (-5.12, 5.12), its=50, mut=0.3, crossp=0.2)
# res2 = genetic_algorithm(sphere, 2, (-5.12, 5.12), 50)
# res3 = pso(sphere, 2, (-5.12, 5.12), 50)
# res4 = PSOL(
#     num_particles=100,
#     num_dims=2,
#     fitness_fn=sphere,
#     num_swarms=25,
#     boundary=(-5.12, 5.12),
#     initialization="uniform",
#     type="min",
# ).run(50)
# res5 = simulated_annealing(sphere, 2, (-5.12, 5.12), 10, 0.1, 1000)


# pos = [res1[-1], res2[-1], res3[-1], res4["best_position"], res5[-1]]
# fitness = [res1[0][-1], res2[0][-1], res3[0][-1], res4["best_fitness"], res5[0][-1]]
# algo = ["DE", "GA", "PSO", "PSOL", "SA"]

# res = [{"algo": a, "pos": p, "fitness": f} for a, p, f in zip(algo, pos, fitness)]
# res = sorted(res, key = lambda o: abs(o["fitness"] - 0))

# pprint(res, indent=2)
