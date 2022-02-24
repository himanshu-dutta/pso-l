from copy import deepcopy
from typing import Callable, Tuple, Union

import numpy as np
from tqdm import tqdm

from .utils import normal_initialization, uniform_initialization


class Particle:
    def __init__(
        self,
        num_dims: int,
        boundary: Tuple[Union[int, float], Union[int, float]],
        global_boundary: Tuple[Union[int, float], Union[int, float]],
        initialization: str = "normal",
    ):

        self.position = (
            normal_initialization(1, num_dims, boundary)
            if initialization == "normal"
            else uniform_initialization(1, num_dims, boundary)
        )
        self.velocity = (
            normal_initialization(1, num_dims, boundary)
            if initialization == "normal"
            else uniform_initialization(1, num_dims, boundary)
        )

        self.global_boundary = global_boundary

        self.update_best_position(True)

    def update_best_position(self, update: bool):
        if update:
            self.best_position = deepcopy(self.position)

    def update(
        self,
        gbest: np.ndarray,
        ggbest: np.ndarray,
        c1: float,
        c2: float,
        c3: float,
        r1: float,
        r2: float,
        r3: float,
        w: float,
    ):

        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (gbest - self.position)
            + c3 * r3 * (ggbest - self.position)
        )

        self.position += self.velocity

        self.position = np.where(
            self.position < self.global_boundary[0],
            self.global_boundary[0],
            self.position,
        )
        self.position = np.where(
            self.position > self.global_boundary[1],
            self.global_boundary[1],
            self.position,
        )


class Swarm:
    def __init__(
        self,
        num_particles: int,
        num_dims: int,
        fitness_fn: Callable,
        boundary: Tuple[Union[int, float], Union[int, float]],
        global_boundary: Tuple[Union[int, float], Union[int, float]],
        initialization: str = "normal",
        type: str = "min",
    ):

        self.particles = [
            Particle(
                num_dims,
                boundary,
                global_boundary=global_boundary,
                initialization=initialization,
            )
            for _ in range(num_particles)
        ]

        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.reverse = True if type == "max" else False

        self.update_gbest()

    def update_gbest(self):

        self.particles = sorted(
            self.particles,
            key=lambda particle: self.fitness_fn(particle.best_position),
            reverse=self.reverse,
        )

        self.gbest = deepcopy(self.particles[0].best_position)

    def update(
        self,
        ggbest: np.ndarray,
        c1: float,
        c2: float,
        c3: float,
        r1: float,
        r2: float,
        r3: float,
        w: float,
    ):
        for particle in self.particles:

            particle.update(
                gbest=self.gbest,
                ggbest=ggbest,
                c1=c1,
                c2=c2,
                c3=c3,
                r1=r1,
                r2=r2,
                r3=r3,
                w=w,
            )

            condition = (
                particle.position
                <= np.full_like(particle.position, fill_value=self.boundary[0])
            ) | (
                particle.position
                > np.full_like(particle.position, fill_value=self.boundary[1])
            )

            if self.reverse:
                if self.fitness_fn(particle.position) > self.fitness_fn(ggbest):
                    ggbest = np.where(condition, particle.position, ggbest)
                else:
                    particle.position += np.where(
                        condition, c2 * r2 * (ggbest - particle.position), 0.0
                    )
            else:
                if self.fitness_fn(particle.position) < self.fitness_fn(ggbest):
                    ggbest = np.where(condition, particle.position, ggbest)
                else:
                    particle.position += np.where(
                        condition, c2 * r2 * (ggbest - particle.position), 0.0
                    )

            # updation of particle best position
            if self.reverse:
                particle.update_best_position(
                    self.fitness_fn(particle.position)
                    > self.fitness_fn(particle.best_position)
                )
            else:
                particle.update_best_position(
                    self.fitness_fn(particle.position)
                    < self.fitness_fn(particle.best_position)
                )

        self.update_gbest()


class PSOL:
    def __init__(
        self,
        num_particles: int,
        num_dims: int,
        fitness_fn: Callable,
        num_swarms: int,
        boundary: Tuple[Union[int, float], Union[int, float]],
        initialization: str = "normal",
        type: str = "min",
    ):
        assert num_swarms < num_particles, ValueError(
            "The number of swarms shouldn't be more than the number of particles."
        )

        step_size = (boundary[1] - boundary[0]) / num_swarms
        particles_per_swarm = num_particles // num_swarms

        self.swarms = []
        last_boundary_start = boundary[0]

        for _ in range(num_swarms):
            self.swarms.append(
                Swarm(
                    num_particles=particles_per_swarm,
                    num_dims=num_dims,
                    fitness_fn=fitness_fn,
                    boundary=[last_boundary_start, last_boundary_start + step_size],
                    global_boundary=boundary,
                    initialization=initialization,
                    type=type,
                )
            )
            last_boundary_start += step_size

        self.fitness_fn = fitness_fn
        self.reverse = True if type == "max" else False

        (
            self.c1,
            self.c2,
            self.c3,
            self.r1,
            self.r2,
            self.r3,
            self.w,
        ) = np.random.rand(7).tolist()

        self.update_ggbest()

    def update_ggbest(self):

        self.swarms = sorted(
            self.swarms,
            key=lambda swarm: self.fitness_fn(swarm.gbest),
            reverse=self.reverse,
        )
        self.ggbest = deepcopy(self.swarms[0].gbest)

    def update(self):
        for swarm in self.swarms:
            swarm.update(
                ggbest=self.ggbest,
                c1=self.c1,
                c2=self.c2,
                c3=self.c3,
                r1=self.r1,
                r2=self.r2,
                r3=self.r3,
                w=self.w,
            )

        self.update_ggbest()

    def run(self, epochs) -> dict:
        history = []
        range_epochs = tqdm(range(epochs))

        for epoch in range_epochs:
            range_epochs.set_description(f"Epoch [{epoch+1} / {epochs}]")
            self.update()
            history.append((self.ggbest.tolist(), self.fitness_fn(self.ggbest)))
            range_epochs.set_postfix_str(f"score: {history[-1][1]}")

        self.update_ggbest()

        return {
            "best_position": self.ggbest.tolist(),
            "best_fitness": self.fitness_fn(self.ggbest),
            "history": history,
        }
