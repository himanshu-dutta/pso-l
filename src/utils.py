import numpy as np
from typing import Tuple, Union


def uniform_initialization(
    num_objects: int,
    num_dims: int,
    boundary: Tuple[Union[int, float], Union[int, float]],
):
    object = (
        np.random.uniform(low=0, high=1, size=(num_objects, num_dims))
        if num_objects > 1
        else np.random.uniform(low=0, high=1, size=(num_dims,))
    )
    object_min, object_max = 0.0, 1.0  # uniform distribution

    object -= object_min
    object *= boundary[1] - boundary[0]
    object /= object_max - object_min
    object += boundary[0]

    return object


def normal_initialization(
    num_objects: int,
    num_dims: int,
    boundary: Tuple[Union[int, float], Union[int, float]],
):
    object = (
        np.random.normal(size=(num_objects, num_dims))
        if num_objects > 1
        else np.random.normal(size=(num_dims,))
    )
    object_min, object_max = (
        object.min().item(),
        object.max().item(),
    )  # normal distribution

    object -= object_min
    object *= boundary[1] - boundary[0]
    object /= object_max - object_min + 1e-5
    object += boundary[0]

    return object
