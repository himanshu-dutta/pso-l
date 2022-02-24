import numpy as np

# ======================
#   STOCK FUNCTIONS
# ======================


def func1(x: np.ndarray):
    return ((-10 * np.sin(x - 5)) / np.abs(x + 2)).sum(-1)


def func2(x: np.ndarray):
    return ((x - 3) ** 2).sum(-1)


def func3(x: np.ndarray):
    return (10 + x ** 2 - 8 * np.cos(2 * np.pi * x)).sum(-1)


def func4(x: np.ndarray):
    return 2 * ((x ** 2) + np.exp(x ** 2)).sum(-1)


def func5(x: np.ndarray):
    return (-np.cos(x)).sum(-1)


# ======================
#   MANY LOCAL MINIMA
# ======================


def cross_in_tray(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]
    return (
        -0.0001
        * (
            np.abs(
                np.sin(x1)
                * np.sin(x2)
                * np.exp(np.abs(100 - (np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)))
            )
            + 1
        )
        ** 0.1
    )


def drop_wave(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]
    return -(
        (1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))) / (0.5 * (x1 ** 2 + x2 ** 2) + 2)
    )


# ======================
#   BOWL SHAPED
# ======================


def bohachevsky(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]
    return (
        x1 ** 2
        + 2 * x2 ** 2
        - 0.3 * np.cos(3 * np.pi * x1)
        - 0.4 * np.cos(4 * np.pi * x2)
        + 0.7
    )


def sphere(x: np.ndarray):
    return np.sum(x ** 2, axis=-1)


# ======================
#   PLATE SHAPED
# ======================


def booth(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def zakharov(x: np.ndarray):
    x = x.reshape((-1,))
    i = np.arange(1, len(x) + 1)
    a = (0.5 * i * x).sum()

    return (x ** 2).sum(axis=-1) + a ** 2 + a ** 4


# ======================
#   VALLEY SHAPED
# ======================


def three_hump_camel(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]

    return 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2


def dixon_prince(x: np.ndarray):
    x = x.reshape((-1,))
    i = np.arange(2, len(x) + 1)

    xi = x[:-1]
    xi_1 = x[1:]

    return (x[0] - 1) ** 2 + (i * (2 * xi ** 2 - xi_1) ** 2).sum(axis=-1)


# ======================
#   STEEP RIDGES/DROPS
# ======================


def easom(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]

    return -(np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2))


def michalewicz(x: np.ndarray):
    if len(x.shape) == 1:
        i = np.arange(1, len(x) + 1)

    else:
        i = np.array([list(range(1, x.shape[-1] + 1))] * x.shape[0])

    return -(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** 20).sum(axis=-1)


# ======================
#   OTHERS
# ======================


def beale(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]

    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * (x2 ** 2)) ** 2
        + (2.625 - x1 + x1 * (x2 ** 3)) ** 2
    )


def goldstein_price(x: np.ndarray):
    x = x.reshape((-1,))
    x1, x2 = x[0], x[1]

    return (
        1
        + (x1 + x2 + 1) ** 2
        * (19 - 14 * x1 + 3 * (x1 ** 2) - 14 * x2 + 6 * x1 * x2 + 3 * (x2 ** 2))
    ) * (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (18 - 2 * x1 + 12 * (x1 ** 2) + 48 * x2 - 36 * x1 * x2 + 27 * (x2 ** 2))
    )
