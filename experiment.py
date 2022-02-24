import json
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from src.pso import PSOL

from compete import genetic_algorithm, pso, differential_evolution, simulated_annealing
from functions import general_functions, two_dims_functions, d_dims_functions


def contour_plot_experiment(config, save_dir_base=Path("results")):
    boundary = config["boundary"]
    func = config["function"]

    save_path = save_dir_base / config["name"]
    save_path.mkdir(parents=True, exist_ok=True)

    psol = PSOL(
        num_particles=config["num_particles"],
        num_dims=2,
        fitness_fn=func,
        boundary=boundary,
        num_swarms=24,
        type="min",
    )
    result = psol.run(epochs=config["epochs"])

    x = np.array([o[0][0] for o in result["history"]])
    y = np.array([o[0][1] for o in result["history"]])

    #   ================================================
    #   CONTOUR PLOT WITH TRAVERSAL FOR GGBEST

    i1 = np.arange(min(x) - 1, max(x) + 1, 0.01)
    i2 = np.arange(min(y) - 1, max(y) + 1, 0.01)

    x1m, x2m = np.meshgrid(i1, i2)
    fm = (
        0.2
        + x1m ** 2
        + x2m ** 2
        - 0.1 * np.cos(6.0 * 3.1415 * x1m)
        - 0.1 * np.cos(6.0 * 3.1415 * x2m)
    )

    plt.figure()
    CS = plt.contour(x1m, x2m, fm)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("dim1")
    plt.ylabel("dim2")

    plt.plot(x, y)
    plt.quiver(
        x[:-1],
        y[:-1],
        x[1:] - x[:-1],
        y[1:] - y[:-1],
        scale_units="xy",
        angles="xy",
        scale=1,
        linewidths=100,
    )
    plt.savefig(str(save_path / "ggbest_traversal.png"), format="png", dpi=1200)
    plt.clf()

    #   ################################################


def fitness_v_swarms_experiment(config, save_dir_base=Path("results")):
    boundary = config["boundary"]
    func = config["function"]
    num_particles = config["num_particles"]
    list_num_swarms = list(range(2, config["num_particles"], 20))

    save_path = save_dir_base / config["name"]
    save_path.mkdir(parents=True, exist_ok=True)

    fitness_v_swarms = []

    for num_swarms in list_num_swarms:
        psol = PSOL(
            num_particles=num_particles,
            num_dims=2,
            fitness_fn=func,
            boundary=boundary,
            num_swarms=num_swarms,
            type="min",
        )
        result = psol.run(epochs=config["epochs"])
        fitness_v_swarms.append(result["best_fitness"])

    x = list_num_swarms
    y = fitness_v_swarms

    #   ================================================
    #   LINE PLOT Fitness vs Num of Swarms

    plt.plot(x, y)
    plt.xlabel("number of swarms")
    plt.ylabel("fitness")

    plt.savefig(str(save_path / "fitness_vs_swarms.png"), format="png", dpi=1200)
    plt.clf()

    #   ################################################


def fitness_v_epoch_experiment(config, save_dir_base=Path("results")):

    boundary = config["boundary"]
    func = config["function"]
    epochs = config["epochs"]
    num_particles = config["num_particles"]
    best_fitness = config["best_fitness"]

    save_path = save_dir_base / config["name"]
    save_path.mkdir(parents=True, exist_ok=True)

    psol = PSOL(
        num_particles=num_particles,
        num_dims=2,
        fitness_fn=func,
        boundary=boundary,
        num_swarms=24,
        type="min",
    )

    psol_results = psol.run(epochs=epochs)
    psol_results, psol_pos = [o[1] for o in psol_results["history"]], psol_results[
        "best_position"
    ]

    ga_results, ga_pos = genetic_algorithm(
        func,
        dims=2,
        bounds=boundary,
        max_iter=epochs,
        num_pop=num_particles,
    )
    pso_results, pso_pos = pso(
        func,
        dims=2,
        bounds=boundary,
        max_iter=epochs,
        num_pop=num_particles,
    )
    de_results, de_pos = differential_evolution(
        func,
        dims=2,
        bounds=boundary,
        popsize=num_particles,
        its=epochs,
    )

    sa_results, sa_pos = simulated_annealing(
        func,
        dims=2,
        bounds=boundary,
        max_iter=epochs,
    )

    fitness_vs_epochs_df = pd.DataFrame(
        {
            "fitness": psol_results
            + pso_results
            + ga_results
            + de_results
            + sa_results,
            "epochs": list(range(epochs)) * 5,
            "Algorithm": (
                ["PSO-L"] * epochs
                + ["PSO"] * epochs
                + ["GA"] * epochs
                + ["DE"] * epochs
                + ["SA"] * epochs
            ),
        }
    )

    #   ================================================
    #   FITNESS VS EPOCHS PLOT
    fitness_vs_epoch_plt = sns.lineplot(
        data=fitness_vs_epochs_df,
        x="epochs",
        y="fitness",
        hue="Algorithm",
    )
    fitness_vs_epoch_plt.get_figure().savefig(
        str(save_path / "fitness_vs_epoch.png"),
        format="png",
        dpi=1200,
    )
    plt.clf()

    #   ################################################

    #   ================================================
    #   RESULT DUMP

    algo_results = [
        {"algo": "ga", "fitness": ga_results[-1], "position": ga_pos},
        {"algo": "pso", "fitness": pso_results[-1], "position": pso_pos},
        {"algo": "psol", "fitness": psol_results[-1], "position": psol_pos},
        {"algo": "de", "fitness": de_results[-1], "position": de_pos},
        {"algo": "sa", "fitness": sa_results[-1], "position": sa_pos},
    ]
    algo_results = sorted(algo_results, key=lambda o: abs(o["fitness"] - best_fitness))

    results = {
        "best_fitness": best_fitness,
        "results": algo_results,
    }

    with open(save_path / "fitness_epochs.json", "w") as fl:
        json.dump(results, fl, indent=4)

    #   ################################################


def fitness_v_dims_experiment(
    config, dims_list=[2, 3, 4, 5], save_dir_base=Path("results")
):

    boundary = config["boundary"]
    func = config["function"]
    epochs = config["epochs"]
    num_particles = config["num_particles"]
    best_fitness = config["best_fitness"]

    save_path = save_dir_base / config["name"]
    save_path.mkdir(parents=True, exist_ok=True)

    # psom
    psol_dims = []

    for dims in dims_list:

        psol = PSOL(
            num_particles=num_particles,
            num_dims=dims,
            fitness_fn=func,
            boundary=boundary,
            num_swarms=24,
            type="min",
        )

        psol_results = psol.run(epochs=epochs)
        psol_results, _ = [o[1] for o in psol_results["history"]], psol_results[
            "best_position"
        ]

        psol_dims.append(round(psol_results[-1], 4))

    # ga
    ga_dims = []

    for dims in dims_list:
        ga_results, _ = genetic_algorithm(
            func,
            dims=dims,
            bounds=boundary,
            max_iter=epochs,
            num_pop=num_particles,
        )

        ga_dims.append(ga_results[-1])

    # pso
    pso_dims = []

    for dims in dims_list:
        pso_results, _ = pso(
            func,
            dims=dims,
            bounds=boundary,
            max_iter=epochs,
            num_pop=num_particles,
        )
        pso_dims.append(pso_results[-1])

    # de
    de_dims = []

    for dims in dims_list:
        de_results, _ = differential_evolution(
            func,
            dims=dims,
            bounds=boundary,
            popsize=num_particles,
            its=epochs,
        )

        de_dims.append(de_results[-1])

    # sa
    sa_dims = []

    for dims in dims_list:
        sa_results, _ = simulated_annealing(
            func,
            dims=dims,
            bounds=boundary,
            max_iter=epochs,
        )

        sa_dims.append(sa_results[-1])

    fitness_vs_dims_df = pd.DataFrame(
        {
            "fitness": psol_dims + pso_dims + ga_dims + de_dims + sa_dims,
            "epochs": list(dims_list) * 5,
            "Algorithm": (
                ["PSO-L"] * len(dims_list)
                + ["PSO"] * len(dims_list)
                + ["GA"] * len(dims_list)
                + ["DE"] * len(dims_list)
                + ["SA"] * len(dims_list)
            ),
        }
    )

    #   ================================================
    #   FITNESS VS EPOCHS PLOT
    fitness_vs_dims_plt = sns.lineplot(
        data=fitness_vs_dims_df,
        x="epochs",
        y="fitness",
        hue="Algorithm",
    )
    fitness_vs_dims_plt.get_figure().savefig(
        str(save_path / "fitness_vs_dims.png"),
        format="png",
        dpi=1200,
    )
    plt.clf()

    #   ################################################

    #   ================================================
    #   RESULT DUMP

    results = {
        "ga": {dim: fit for dim, fit in zip(dims_list, ga_dims)},
        "pso": {dim: fit for dim, fit in zip(dims_list, pso_dims)},
        "psol": {dim: fit for dim, fit in zip(dims_list, psol_dims)},
        "de": {dim: fit for dim, fit in zip(dims_list, de_dims)},
    }

    with open(save_path / "fitness_dims.json", "w") as fl:
        json.dump(results, fl, indent=4)
    #   ################################################


if __name__ == "__main__":
    for func_config in general_functions:
        contour_plot_experiment(func_config)
        fitness_v_epoch_experiment(func_config)
        fitness_v_swarms_experiment(func_config)

    for func_config in two_dims_functions:
        contour_plot_experiment(func_config)
        fitness_v_epoch_experiment(func_config)
        fitness_v_swarms_experiment(func_config)

    for func_config in d_dims_functions:
        fitness_v_dims_experiment(func_config, dims_list=[2, 3, 4, 5, 6, 7, 8, 9, 10])
