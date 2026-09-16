"""Population distribution over twin parameters, for large-scale experiments.

Pools fitted people into one Normal over the global parameters: the spread
between people plus each person's own posterior uncertainty, shrunk toward the
prior covariance when there are few people. Synthetic people drawn from it are
run over real people's recorded days (their context and fitted meals), so a
CR/ISF sweep covers plausible bodies in plausible lives.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from t1d_twin.data import PersonTimeline
from t1d_twin.experiment import Arm, run_settings_experiment
from t1d_twin.fit import TwinFit
from t1d_twin.model import DTYPE

PRIOR_PSEUDO_PEOPLE = 5.0


@dataclass
class TwinPopulation:
    param_names: list[str]
    mean: list[float]
    cov: list[list[float]]
    n_people: int

    def sample(self, n: int, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        return torch.tensor(rng.multivariate_normal(self.mean, self.cov, size=n), dtype=DTYPE)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TwinPopulation":
        return cls(**json.loads(Path(path).read_text()))


def _posterior_cov(fit: TwinFit) -> np.ndarray:
    if fit.scale_tril:
        L = np.asarray(fit.scale_tril)
        return L @ L.T
    return np.diag(np.square(fit.sd))


def build_population(fits: list[TwinFit]) -> TwinPopulation:
    """Between-person + mean within-person covariance, shrunk toward the prior."""
    if not fits:
        raise ValueError("need at least one fitted person")
    names = fits[0].param_names
    locs = np.array([f.loc for f in fits])
    prior_mean = np.array(fits[0].prior_mean)
    prior_cov = np.diag(np.square(fits[0].prior_sd))
    n = len(fits)
    within = np.mean([_posterior_cov(f) for f in fits], axis=0)
    between = np.cov(locs, rowvar=False) if n > 1 else np.zeros_like(prior_cov)
    w = n / (n + PRIOR_PSEUDO_PEOPLE)
    mean = w * locs.mean(axis=0) + (1 - w) * prior_mean
    cov = w * (between + within) + (1 - w) * prior_cov
    cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(len(names))
    return TwinPopulation(names, mean.tolist(), cov.tolist(), n)


def run_population_experiment(
    population: TwinPopulation,
    scenarios: list[tuple[TwinFit, PersonTimeline]],
    arms: list[Arm],
    *,
    n_people: int = 64,
    days: tuple[int, int] | None = None,
    aid: bool = False,
    seed: int = 0,
) -> dict:
    """Synthetic people from ``population`` living the recorded days in ``scenarios``.

    People are split evenly across scenarios; each scenario runs as one
    batched, paired rollout. A synthetic person keeps the scenario person's
    base adult (its structural dynamics) and gets new global parameters.
    """
    draws = population.sample(n_people, seed)
    per = np.array_split(np.arange(n_people), len(scenarios))
    runs = []
    for (fit, tl), idx in zip(scenarios, per):
        if len(idx) == 0:
            continue
        runs.append(run_settings_experiment(fit, tl, arms, days=days, aid=aid, seed=seed, globals_override=draws[idx]))
    return {"n_people": n_people, "population_from_n_fitted": population.n_people, "runs": runs}
