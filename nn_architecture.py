import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import seaborn as sns
import matplotlib.pyplot as plt

from utils import *
from src.pso import PSOL
from compete import genetic_algorithm, simulated_annealing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(num_ins, num_outs=1, hidden_units=[4, 4]):
    model = nn.Sequential(
        nn.Linear(num_ins, int(hidden_units[0])),
        nn.Linear(int(hidden_units[0]), int(hidden_units[1])),
        nn.Linear(int(hidden_units[1]), num_outs),
    )

    return model.to(device)


def train(model, epochs, train_dl, val_dl, thres=0.7):
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    for _ in tqdm(range(epochs), leave=False):
        for x, y in train_dl:

            x, y = x.to(device), y.to(device)
            preds = model(x)
            preds = preds.reshape((-1,))

            loss = F.binary_cross_entropy_with_logits(preds, y)
            loss.backward()
            opt.step()

    accs = []
    for x, y in val_dl:

        x, y = x.to(device), y.to(device)
        preds = model(x)
        preds = torch.sigmoid(preds)
        preds = (preds > thres).float()
        preds = preds.reshape((-1,))

        acc = torch.abs(preds - y).mean().item()
        accs.append(acc)

    acc = sum(accs) / len(accs)

    return acc


def get_fitness(train_dl, val_dl, epochs, num_ins, num_outs=1, thres=0.7):
    def _wrapper(hidden_units):
        hidden_units = list(map(int, hidden_units))
        model = get_model(num_ins, num_outs, hidden_units)
        acc = train(model, epochs, train_dl, val_dl, thres)
        return 1 - acc

    _wrapper.__name__ = _wrapper.__qualname__ = "nn_fitness"
    return _wrapper


def run(
    dataset,
    epochs=10,
    batch_size=32,
    save_dir_base=Path("nn_architecture"),
):
    X, y = load_dataset(dataset)

    num_ins = X.shape[1]

    X, y = (
        [torch.tensor(x).float() for x in X.tolist()],
        [torch.tensor(y_).float() for y_ in y.tolist()],
    )
    ds = list(zip(X, y))

    train_sz = int(len(ds) * 0.8)
    val_sz = len(ds) - train_sz
    train_ds, val_ds = random_split(ds, [train_sz, val_sz])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=6)

    fitness = get_fitness(train_dl, val_dl, epochs, num_ins)
    bounds = [5, 20]
    population_size = 1000
    opt_epochs = 20

    psol = PSOL(
        num_particles=population_size,
        num_dims=2,
        fitness_fn=fitness,
        num_swarms=24,
        boundary=bounds,
    )

    psol_results = psol.run(opt_epochs)
    psol_results, psol_pos = [o[1] for o in psol_results["history"]], psol_results[
        "best_position"
    ]
    psol_acc = 1 - psol_results[-1]
    psol_pos = list(map(int, psol_pos))

    # GA
    ga_results, ga_pos = genetic_algorithm(
        fitness,
        dims=2,
        bounds=bounds,
        max_iter=opt_epochs,
        num_pop=population_size,
    )

    ga_acc = 1 - ga_results[-1]
    ga_pos = list(map(int, ga_pos))

    # DE
    sa_results, sa_pos = simulated_annealing(
        fitness,
        dims=2,
        bounds=bounds,
        max_iter=opt_epochs,
    )

    sa_acc = 1 - sa_results[-1]
    sa_pos = list(map(int, sa_pos))

    #   ================================================
    #   FITNESS VS EPOCHS PLOT
    save_path = save_dir_base / dataset
    save_path.mkdir(parents=True, exist_ok=True)

    fitness_vs_epochs_df = pd.DataFrame(
        {
            "fitness": psol_results + sa_results + ga_results,
            "epochs": list(range(opt_epochs)) * 3,
            "Algorithm": (
                ["PSO-L"] * opt_epochs + ["SA"] * opt_epochs + ["GA"] * opt_epochs
            ),
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
        {"algo": "PSOL", "accuracy": psol_acc, "hidden_units": psol_pos},
        {"algo": "SA", "accuracy": sa_acc, "hidden_units": sa_pos},
        {"algo": "GA", "accuracy": ga_acc, "hidden_units": ga_pos},
    ]
    res = sorted(res, key=lambda o: o["accuracy"], reverse=True)
    results = {
        "dataset": dataset,
        "results": res,
    }

    with open(save_path / "results.json", "w") as fl:
        json.dump(results, fl, indent=4)


if __name__ == "__main__":
    DS = ["Titanic", "Breast Cancer", "Diabetes"]

    for ds in DS:
        run(ds)
