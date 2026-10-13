"""Authored interval-torque excitations; reject limits instead of clipping."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from mujoco_sysid_lab.experiment_config import StudyConfig


def generate_control(
    study: StudyConfig, trajectory_id: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Generate canonical grid/control and sampled energy/RMS diagnostics.

    Controls have N samples; only the first N-1 drive intervals. The terminal
    value is zero and unused. Multisine phases and chirp phase start at active
    time zero (the end of the registered zero-control prefix). PRBS randomness
    is independent of measurement noise and uses the serialized SeedSequence.
    """
    if trajectory_id not in study.trajectories:
        raise ValueError(f"unregistered trajectory ID: {trajectory_id}")
    config = study.trajectories[trajectory_id]
    dt = study.timestep_s
    intervals = int(round(config.duration_s / dt))
    times = np.arange(intervals + 1, dtype=np.float64) * dt
    torque = np.zeros(intervals + 1, dtype=np.float64)
    prefix_intervals = int(round(study.zero_control_prefix_s / dt))
    active_indices = np.arange(prefix_intervals, intervals)
    active_times = (active_indices - prefix_intervals) * dt
    active_duration = config.duration_s - study.zero_control_prefix_s
    settings = config.settings
    details: dict[str, Any] = {}
    if config.kind == "multisine":
        active = sum(
            amplitude * np.sin(2*np.pi*frequency*active_times + phase)
            for frequency, amplitude, phase in zip(
                settings["frequencies_hz"], settings["amplitudes_nm"], settings["phases_rad"],
            )
        )
        scale = _rms_scale(active, settings["active_rms_nm"])
        torque[active_indices] = active * scale
        details.update(frequencies_hz=list(settings["frequencies_hz"]), raw_amplitudes_nm=list(settings["amplitudes_nm"]), phases_rad=list(settings["phases_rad"]), rms_rescale_factor=scale)
    elif config.kind == "prbs":
        block_intervals = int(round(settings["block_duration_s"] / dt))
        block_ids = np.arange(active_indices.size) // block_intervals
        block_count = int(block_ids[-1]) + 1
        seed = np.random.SeedSequence(settings["seed_sequence"])
        rng = np.random.Generator(np.random.PCG64(seed))
        levels = rng.choice(settings["choices"], size=block_count)
        active = levels[block_ids]
        scale = _rms_scale(active, settings["active_rms_nm"])
        torque[active_indices] = active * scale
        details.update(seed_sequence=list(settings["seed_sequence"]), bit_generator="PCG64", block_count=block_count, block_duration_s=settings["block_duration_s"], raw_block_levels=levels.tolist(), scaled_block_levels_nm=(levels*scale).tolist(), last_block_intervals=int(np.sum(block_ids == block_count-1)), rms_rescale_factor=scale)
    elif config.kind == "chirp":
        slope = (settings["end_frequency_hz"] - settings["start_frequency_hz"]) / active_duration
        phase = 2*np.pi*(settings["start_frequency_hz"]*active_times + 0.5*slope*active_times**2)
        envelope = np.ones(active_times.size)
        onset = active_times < settings["onset_s"]
        envelope[onset] = 0.5*(1-np.cos(np.pi*active_times[onset]/settings["onset_s"]))
        torque[active_indices] = settings["amplitude_nm"]*np.sin(phase)*envelope
        details.update(start_frequency_hz=settings["start_frequency_hz"], end_frequency_hz=settings["end_frequency_hz"], sweep_duration_s=active_duration, onset_s=settings["onset_s"], onset_envelope="0.5*(1-cos(pi*active_time/onset_s))", phase_reference="active_time_zero")
    elif config.kind == "pulse":
        pulses = []
        for pulse in settings["pulses"]:
            start = int(round(pulse["start_s"]/dt))
            end = int(round(pulse["end_s"]/dt))
            torque[start:end] = pulse["amplitude_nm"]
            pulses.append(dict(pulse, interval_count=end-start))
        details["pulses"] = pulses
    elif config.kind == "step":
        torque[active_indices] = settings["amplitude_nm"]
        details["step_amplitude_nm"] = settings["amplitude_nm"]
    else:
        raise ValueError(f"unsupported trajectory kind: {config.kind}")
    if not np.isfinite(torque).all():
        raise ValueError(f"{trajectory_id} generated non-finite torque")
    max_abs = float(np.max(np.abs(torque)))
    if max_abs > study.nominal.control_limit_nm:
        raise ValueError(
            f"{trajectory_id} peak torque {max_abs:.12g} N*m exceeds registered "
            f"limit {study.nominal.control_limit_nm:.12g}; no clipping is applied"
        )
    controls = (torque/study.nominal.motor_gain_nm_per_unit)[:, None]
    intervals_torque = torque[:-1]
    active_torque = torque[active_indices]
    diagnostics = {
        "trajectory_id": trajectory_id,
        "numeric_trajectory_id": config.trajectory_id,
        "split": config.split,
        "kind": config.kind,
        "duration_s": config.duration_s,
        "timestep_s": dt,
        "samples": intervals+1,
        "driven_intervals": intervals,
        "zero_control_prefix_s": study.zero_control_prefix_s,
        "active_duration_s": active_duration,
        "active_intervals": active_indices.size,
        "terminal_control_unused": True,
        "terminal_control_value": float(controls[-1, 0]),
        "torque_peak_abs_nm": max_abs,
        "torque_rms_nm": float(np.sqrt(np.mean(intervals_torque**2))),
        "active_torque_rms_nm": float(np.sqrt(np.mean(active_torque**2))),
        "torque_dc_nm": float(np.mean(intervals_torque)),
        "active_torque_dc_nm": float(np.mean(active_torque)),
        "sampled_torque_energy_nm2_s": float(np.sum(intervals_torque**2)*dt),
        "control_units": "motor_control; torque_nm=control*known_gain",
        "energy_rms_scope": "all_driven_intervals_excluding_unused_terminal_sample",
        "reset_count": 1,
        **details,
    }
    times.flags.writeable = False
    controls.flags.writeable = False
    return times, controls, diagnostics


def _rms_scale(active: NDArray[np.float64], target_rms: float) -> float:
    rms = float(np.sqrt(np.mean(active**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("cannot normalize a zero or non-finite excitation RMS")
    return float(target_rms/rms)
