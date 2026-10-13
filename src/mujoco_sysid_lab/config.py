"""Validated, immutable mechanics for the first verification milestone.

The config is a simulation contract, not an estimator interface. The joint
estimator will receive nominal mechanics without the hidden truth object.
"""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mujoco-sysid-lab.milestone1.v1"


def _finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class ParameterBounds:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        _finite_number("bounds.lower", self.lower)
        _finite_number("bounds.upper", self.upper)
        if self.lower >= self.upper:
            raise ValueError("parameter bounds require lower < upper")

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class ModelConfig:
    mass_kg: float
    damping_nm_s_per_rad: float
    timestep_s: float
    link_length_m: float
    link_radius_m: float
    motor_gain_nm_per_unit: float
    control_limit_nm: float
    initial_qpos_rad: float
    initial_qvel_rad_s: float

    def __post_init__(self) -> None:
        for field in fields(self):
            _finite_number(field.name, getattr(self, field.name))
        for name in (
            "mass_kg", "timestep_s", "link_length_m", "link_radius_m",
            "motor_gain_nm_per_unit", "control_limit_nm",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.damping_nm_s_per_rad < 0:
            raise ValueError("damping_nm_s_per_rad must be nonnegative")

    @property
    def inertia_at_com_kg_m2(self) -> tuple[float, float, float]:
        """Uniform-cylinder COM inertia; changing mass scales every axis."""
        transverse = self.mass_kg * (
            self.link_length_m**2 / 12 + self.link_radius_m**2 / 4
        )
        axial = self.mass_kg * self.link_radius_m**2 / 2
        return transverse, transverse, axial


@dataclass(frozen=True)
class LabConfig:
    schema_version: str
    experiment_id: str
    seed: int
    duration_s: float
    ground_truth: ModelConfig
    nominal: ModelConfig
    mass_bounds_kg: ParameterBounds
    damping_bounds_nm_s_per_rad: ParameterBounds

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported config schema: {self.schema_version!r}")
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("experiment_id must be a nonempty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _finite_number("duration_s", self.duration_s)
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self.mass_bounds_kg.lower <= 0:
            raise ValueError("mass bounds must be positive")
        if self.damping_bounds_nm_s_per_rad.lower < 0:
            raise ValueError("damping bounds must be nonnegative")
        for label, model in (("ground_truth", self.ground_truth), ("nominal", self.nominal)):
            if not self.mass_bounds_kg.contains(model.mass_kg):
                raise ValueError(f"{label}.mass_kg is outside mass bounds")
            if not self.damping_bounds_nm_s_per_rad.contains(model.damping_nm_s_per_rad):
                raise ValueError(f"{label}.damping_nm_s_per_rad is outside damping bounds")
            intervals = self.duration_s / model.timestep_s
            if intervals < 1 or not math.isclose(intervals, round(intervals), abs_tol=1e-9):
                raise ValueError("duration_s must be a positive integer multiple of timestep_s")
        fixed_fields = (
            "timestep_s", "link_length_m", "link_radius_m", "motor_gain_nm_per_unit",
            "control_limit_nm", "initial_qpos_rad", "initial_qvel_rad_s",
        )
        for name in fixed_fields:
            if getattr(self.ground_truth, name) != getattr(self.nominal, name):
                raise ValueError(f"ground_truth and nominal must share fixed {name}")


def _construct(cls: type, data: Any, label: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    expected = {field.name for field in fields(cls)}
    if set(data) != expected:
        missing = sorted(expected - set(data))
        unknown = sorted(set(data) - expected)
        raise ValueError(f"{label} fields invalid: missing={missing}, unknown={unknown}")
    return cls(**data)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def load_config(path: Path) -> LabConfig:
    """Load strict versioned JSON; reject unknown keys and invalid mechanics."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(
            stream, parse_constant=_reject_nonfinite_json, object_pairs_hook=_unique_object,
        )
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")
    for name in ("ground_truth", "nominal"):
        if name in raw:
            raw[name] = _construct(ModelConfig, raw[name], name)
    for name in ("mass_bounds_kg", "damping_bounds_nm_s_per_rad"):
        if name in raw:
            raw[name] = _construct(ParameterBounds, raw[name], name)
    return _construct(LabConfig, raw, "config")
