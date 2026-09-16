"""The UVA/Padova 2008 virtual-patient parameters, read from simglucose.

simglucose ships the published parameter table for the 30 virtual subjects
(10 adults, 10 adolescents, 10 children). We read it at import time rather than
copying it, so the numbers stay the ones its authors distribute.

simglucose 0.2.11 imports gym at module load, which newer runtimes no longer
provide; ``_install_gym_shim`` supplies the two names it touches.
"""

from __future__ import annotations

import sys
import types
from functools import lru_cache


def _install_gym_shim() -> None:
    if "gym.envs.registration" in sys.modules:
        return
    gym_mod = types.ModuleType("gym")
    envs_mod = types.ModuleType("gym.envs")
    registration_mod = types.ModuleType("gym.envs.registration")
    registration_mod.register = lambda *args, **kwargs: None
    envs_mod.registration = registration_mod
    gym_mod.envs = envs_mod
    sys.modules.setdefault("gym", gym_mod)
    sys.modules.setdefault("gym.envs", envs_mod)
    sys.modules.setdefault("gym.envs.registration", registration_mod)


@lru_cache(maxsize=1)
def vpatient_frame():
    """The virtual-patient parameter table as a pandas DataFrame."""
    _install_gym_shim()
    import pandas as pd

    try:  # simglucose >= 0.2.7 ships the csv inside the package
        from importlib.resources import files

        path = files("simglucose").joinpath("params/vpatient_params.csv")
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - install problem
        raise RuntimeError("simglucose is required for the virtual-patient parameters (pip install simglucose)") from exc
    with path.open() as fh:
        return pd.read_csv(fh)


def t1d_patient_class():
    """simglucose's own patient model, used as the reference in tests."""
    _install_gym_shim()
    from simglucose.patient.t1dpatient import T1DPatient

    return T1DPatient
