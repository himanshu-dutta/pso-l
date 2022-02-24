import json
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

# model algorithm
from utils import *
from src.pso import PSOL
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from compete import genetic_algorithm, simulated_annealing


def run(
    dataset: str,
    model: str,
    epochs: int = 10,
    save_dir_base=Path("feature_optimization"),
    **kwargs,
):

    opt = "PSO"

    X, y = load_dataset(dataset)

    # setting hyperparameters
    SIZE = 50  # number of particles
    TH = 0.7  # threshold to appect or reject the feature
    MAX = X.shape[1]
    DIM = MAX
    BOUNDS = (0.0, 1.0)

    if model == "DT":
        fitness = set_params(
            DecisionTreeClassifier, X, y, "fit", "predict", TH, **kwargs
        )

    elif model == "NB":
        fitness = set_params(GaussianNB, X, y, "fit", "predict", TH, **kwargs)

    # PSOL
    psol_results = PSOL(
        num_particles=SIZE,
        num_dims=DIM,
        fitness_fn=fitness,
        boundary=BOUNDS,
        num_swarms=20,
        type="min",
    ).run(epochs)
    psol_results, psol_pos = [o[1] for o in psol_results["history"]], psol_results[
        "best_position"
    ]

    psol_best, psol_features = psol_results[-1], psol_pos

    psol_features = [1 if col > TH else 0 for col in psol_features]
    psol_features = (
        np.argwhere(psol_features == np.max(psol_features))
        .reshape(
            -1,
        )
        .tolist()
    )
    psol_acc = 1 - psol_best

    # GA
    ga_results, ga_pos = genetic_algorithm(
        fitness,
        dims=DIM,
        bounds=BOUNDS,
        max_iter=epochs,
        num_pop=SIZE,
    )

    ga_best, ga_features = ga_results[-1], ga_pos

    ga_features = [1 if col > TH else 0 for col in ga_features]
    ga_features = (
        np.argwhere(ga_features == np.max(ga_features))
        .reshape(
            -1,
        )
        .tolist()
    )
    ga_acc = 1 - ga_best

    # PSO
    de_results, de_pos = simulated_annealing(
        fitness,
        dims=DIM,
        bounds=BOUNDS,
        max_iter=epochs,
    )

    de_best, de_features = de_results[-1], de_pos

    de_features = [1 if col > TH else 0 for col in de_features]
    de_features = (
        np.argwhere(de_features == np.max(de_features))
        .reshape(
            -1,
        )
        .tolist()
    )
    de_acc = 1 - de_best

    #   ================================================
    #   FITNESS VS EPOCHS PLOT
    save_path = save_dir_base / (dataset + "_" + model)
    save_path.mkdir(parents=True, exist_ok=True)

    fitness_vs_epochs_df = pd.DataFrame(
        {
            "fitness": psol_results + de_results + ga_results,
            "epochs": list(range(epochs)) * 3,
            "Algorithm": (["GA"] * epochs + ["DE"] * epochs + ["PSO-L"] * epochs),
        }
    )

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

    res = [
        {"algo": "GA", "accuracy": psol_acc, "features": psol_features},
        {"algo": "DE", "accuracy": de_acc, "features": de_features},
        {"algo": "PSOL", "accuracy": ga_acc, "features": ga_features},
    ]
    res = sorted(res, key=lambda o: o["accuracy"], reverse=True)
    results = {
        "dataset": dataset,
        "model": model,
        "results": res,
    }

    with open(save_path / "results.json", "w") as fl:
        json.dump(results, fl, indent=4)


if __name__ == "__main__":
    MODELS = {"DT": {"max_depth": 5}, "NB": {}}
    DS = ["Titanic", "Breast Cancer", "Diabetes"]

    for mdl in MODELS:
        for ds in DS:
            run(ds, mdl, 100, **MODELS[mdl])
