"""Small CPU simulation with explicit state/control timestamp semantics."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from mujoco_sysid_lab.config import ModelConfig


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Trajectory:
    """State and fresh sensor observations at t[k], with u[k] stored beside it.

    u[k] drives [t[k], t[k+1]). The last control is retained for the common
    N-sample storage shape but is unused. qpos and qvel are (N, 1); sensordata
    is (N, 2), ordered [qpos, qvel]. Arrays are owned and read-only.
    """

    timestamps: FloatArray
    controls: FloatArray
    qpos: FloatArray
    qvel: FloatArray
    sensordata: FloatArray


def load_spec(model_path: Path, config: ModelConfig) -> mujoco.MjSpec:
    """Load authored MJCF as MjSpec and stamp the registered mechanics.

    Mass and COM inertia are changed together. Gain, COM, geometry, gravity,
    armature, dry friction, and initial state are explicit fixed mechanics.
    """
    spec = mujoco.MjSpec.from_file(str(model_path))
    link = spec.body("link")
    hinge = spec.joint("hinge")
    geom = spec.geom("link_geometry")
    motor = spec.actuator("torque")
    for kind, name, node in (("body", "link", link), ("joint", "hinge", hinge),
                             ("geom", "link_geometry", geom), ("actuator", "torque", motor)):
        if node is None:
            raise ValueError(f"pendulum model requires named {kind} {name!r}")
    link.mass = config.mass_kg
    link.inertia = config.inertia_at_com_kg_m2
    link.ipos = [0.0, 0.0, -config.link_length_m / 2]
    # MjSpec 3.15 exposes three damping coefficients; the first is linear
    # viscous damping. Higher-order terms are explicitly absent here.
    hinge.damping = [config.damping_nm_s_per_rad, 0.0, 0.0]
    hinge.armature = 0.0
    hinge.frictionloss = 0.0
    geom.pos = [0.0, 0.0, -config.link_length_m / 2]
    geom.size = [config.link_radius_m, config.link_length_m / 2, 0.0]
    motor.gear = [config.motor_gain_nm_per_unit, 0.0, 0.0, 0.0, 0.0, 0.0]
    motor.ctrlrange = [
        -config.control_limit_nm / config.motor_gain_nm_per_unit,
        config.control_limit_nm / config.motor_gain_nm_per_unit,
    ]
    spec.option.timestep = config.timestep_s
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return spec


def _validate_model_contract(model: mujoco.MjModel) -> None:
    """Require the named hinge/sensor layout before assigning physical units."""
    if (model.nq, model.nv, model.nu, model.na, model.nsensor, model.nsensordata) != (1, 1, 1, 0, 2, 2):
        raise ValueError("pendulum model requires nq=nv=nu=1, na=0 and two scalar sensors")
    hinge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    if (model.njnt != 1 or hinge_id != 0
            or model.jnt_type[hinge_id] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("pendulum model requires exactly the named hinge joint")
    for address, (name, sensor_type) in enumerate(
        (("qpos", mujoco.mjtSensor.mjSENS_JOINTPOS), ("qvel", mujoco.mjtSensor.mjSENS_JOINTVEL))
    ):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if (sensor_id < 0 or model.sensor_type[sensor_id] != sensor_type
                or model.sensor_objtype[sensor_id] != mujoco.mjtObj.mjOBJ_JOINT
                or model.sensor_objid[sensor_id] != hinge_id or model.sensor_dim[sensor_id] != 1
                or model.sensor_adr[sensor_id] != address):
            raise ValueError("pendulum sensors must be named jointpos qpos then jointvel qvel on the hinge")


def compile_model(model_path: Path, config: ModelConfig) -> mujoco.MjModel:
    """Compile stamped MjSpec and check the one-DOF sensor/actuator contract."""
    model = load_spec(model_path, config).compile()
    _validate_model_contract(model)
    return model


def simulate(
    model: mujoco.MjModel,
    times: FloatArray,
    controls: FloatArray,
    *,
    initial_qpos: float = 0.0,
    initial_qvel: float = 0.0,
) -> Trajectory:
    """Run deterministic interval controls from the model's explicit reset state.

    mj_step leaves some sensor fields at their pre-integration evaluation;
    mj_forward refreshes them at the new state before that state is recorded.
    No rendering, process noise, measurement noise, or state refitting occurs.
    """
    _validate_model_contract(model)
    timestamps = np.array(times, dtype=np.float64, copy=True)
    inputs = np.array(controls, dtype=np.float64, copy=True)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps must be a one-dimensional array of at least two samples")
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    if inputs.shape != (timestamps.size, model.nu):
        raise ValueError("controls must have shape (N, nu), including one unused terminal control")
    if not np.isfinite(timestamps).all() or not np.isfinite(inputs).all():
        raise ValueError("timestamps and controls must be finite")
    if not np.isfinite([initial_qpos, initial_qvel]).all():
        raise ValueError("initial qpos and qvel must be finite")
    if timestamps[0] != 0.0:
        raise ValueError("timestamps must start at the explicit initial time 0.0")
    expected = np.arange(timestamps.size, dtype=np.float64) * model.opt.timestep
    if not np.allclose(timestamps, expected, rtol=0, atol=1e-12):
        raise ValueError("timestamps must be strictly increasing and match the simulation timestep")
    lower, upper = model.actuator_ctrlrange[0]
    if lower >= upper:
        raise ValueError("model must register a nonempty actuator control range")
    if (inputs < lower).any() or (inputs > upper).any():
        raise ValueError("control exceeds the registered motor torque limit; no clipping is applied")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.time = 0.0
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = inputs[0]
    mujoco.mj_forward(model, data)
    qpos = np.empty((timestamps.size, model.nq), dtype=np.float64)
    qvel = np.empty((timestamps.size, model.nv), dtype=np.float64)
    sensors = np.empty((timestamps.size, model.nsensordata), dtype=np.float64)
    for index in range(timestamps.size):
        qpos[index] = data.qpos
        qvel[index] = data.qvel
        sensors[index] = data.sensordata
        if index + 1 < timestamps.size:
            data.ctrl[:] = inputs[index]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    if not all(np.isfinite(array).all() for array in (qpos, qvel, sensors)):
        raise RuntimeError("simulation produced non-finite state or sensor values")
    for array in (timestamps, inputs, qpos, qvel, sensors):
        array.flags.writeable = False
    return Trajectory(timestamps, inputs, qpos, qvel, sensors)
