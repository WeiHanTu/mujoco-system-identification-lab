"""Mocked orchestration contracts; these tests do not run evidence fits.

All numeric worker outputs below are explicitly test fixtures. They exercise
selection, cache integrity, and failure persistence rather than optimizer or
scientific recovery behavior.
"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from mujoco_sysid_lab.dataset import TrainingDataset, TrainingSequence
from mujoco_sysid_lab.experiment_config import config_to_dict, load_ground_truth, load_study_config
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, write_json
from mujoco_sysid_lab import study_runner as runner


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "pendulum.xml"
FREE = ("mass_kg", "damping_nm_s_per_rad")


@pytest.fixture(scope="module")
def study():
    return load_study_config(ROOT / "configs/evidence_v1.json")


@pytest.fixture(scope="module")
def provenance():
    return collect_provenance(ROOT, ROOT / "configs/evidence_v1.json", MODEL,
                              [".venv/bin/python", "-m", "pytest", "tests/test_study_runner.py"])


@pytest.fixture
def training():
    # Sanitized tiny orchestration fixture, with no simulator/optimizer calls.
    times = np.arange(5, dtype=np.float64) * .002
    sequences = tuple(TrainingSequence(name, numeric_id, times.copy(), np.zeros((5, 1)),
                     np.zeros((5, 1)), np.zeros((5, 1)), np.zeros(3))
                     for numeric_id, name in enumerate(("train_multisine", "train_prbs")))
    return TrainingDataset(101, sequences)


def candidate(start_id="S0", objective=1., *, cell="B", seed=101, **updates):
    return {"test_fixture": True, "fit_id": f"{cell}_seed{seed}_{start_id}", "cell": cell,
            "seed": seed, "start_id": start_id, "optimization_invoked": True,
            "optimization_returned": True, "optimizer_success": True,
            "optimized_vector": [1.2, .15], "final_objective": objective,
            "free_parameters": list(FREE), "status": "completed", **updates}


def test_selection_includes_nonconverged_report_failed_fit_and_ignores_truth():
    rows = [candidate("S0", 9., heldout_rmse=0., physical_error=0.),
            candidate("S1", 3., optimizer_success=False, status="post_optimization_failed",
                      heldout_rmse=100., physical_error=100.), candidate("S2", 5.)]
    selected = runner.select_rows(rows, FREE)
    assert selected["selected_fit_id"] == "B_seed101_S1"
    assert selected["selected_final_objective"] == 3.
    assert selected["physical_truth_evaluated"] is False
    assert selected["heldout_evaluated"] is False


def test_exact_tie_uses_registered_order_independent_of_input_order():
    rows = [candidate("S2", 3.), candidate("S1", 3.), candidate("S0", 3.)]
    assert runner.select_rows(rows, FREE)["selected_fit_id"] == "B_seed101_S0"
    rows[0]["final_objective"] = np.nextafter(3., -np.inf)
    assert runner.select_rows(rows, FREE)["selected_fit_id"] == "B_seed101_S2"


@pytest.mark.parametrize("updates", [
    {"optimization_returned": False}, {"final_objective": None},
    {"final_objective": -1.}, {"final_objective": np.inf},
    {"final_objective": np.nan}, {"final_objective": True},
    {"optimized_vector": [1.2]}, {"optimized_vector": [np.nan, .15]},
    {"optimized_vector": [1.2, np.inf]}, {"optimized_vector": [True, .15]},
    {"optimized_vector": ["bad", .15]}, {"optimization_returned": "yes"},
])
def test_invalid_numerical_candidates_are_not_selected(updates):
    selection = runner.select_rows([candidate(**updates)], FREE)
    assert selection["selected_fit_id"] is None
    assert selection["selected_vector"] is None
    json.dumps(selection, allow_nan=False)


@pytest.mark.parametrize("rows,free", [
    ([], FREE), ([candidate(), candidate()], FREE),
    ([candidate("S9")], FREE), ([candidate()], ()),
    ([candidate()], ("mass_kg", "mass_kg")),
    ([candidate(), candidate("S1", seed=102)], FREE),
    ([candidate(), candidate("S1", cell="weak")], FREE),
    ([candidate(fit_id="E_seed101_S0")], FREE),
    ([candidate(free_parameters=list(reversed(FREE)))], FREE),
])
def test_selection_rejects_invalid_group_or_start_identity(rows, free):
    with pytest.raises(ValueError):
        runner.select_rows(rows, free)


def test_delay_only_single_start_is_valid():
    row = candidate(cell="C_delay_only", optimized_vector=[.024], free_parameters=["delay_s"])
    assert runner.select_rows([row], ("delay_s",))["selected_vector"] == [.024]


@pytest.fixture
def cached_root(tmp_path, training, study, provenance, monkeypatch):
    # A complete labeled mock cache, using the real registered config/model
    # bytes and current test-command provenance, not invented source IDs.
    root = tmp_path / "mock_cache"
    # Cache semantic tests use labeled mock provenance; original Git source
    # verification is exercised independently below with real cloned fixtures.
    monkeypatch.setattr(runner, "verify_cache_source", lambda *args: {"test_fixture": True})
    (root / "configs").mkdir(parents=True)
    (root / "model").mkdir()
    (root / "configs/evidence_v1.json").write_bytes((ROOT / "configs/evidence_v1.json").read_bytes())
    (root / "model/pendulum.xml").write_bytes(MODEL.read_bytes())
    source = root / "artifacts/milestone4/20261013_seed101"
    indexed = []
    for start_id in ("S0", "S1", "S2"):
        directory = source / "fits" / f"B_seed101_{start_id}"
        directory.mkdir(parents=True)
        row = candidate(start_id)
        row.update(starting_values={name: getattr(study.starts[start_id], name) for name in FREE},
                   initial_physics=config_to_dict(replace(study.nominal,
                        **{name: getattr(study.starts[start_id], name) for name in FREE})),
                   input_array_sha256=runner.training_hashes(training),
                   training_ids=[item.trajectory_name for item in training.sequences],
                   optimizer_options=config_to_dict(study.optimizer),
                   resolved_optimizer_options={"method": "trf", "loss": "linear", "max_iters": 200,
                         "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8, "diff_step": 1e-5,
                         "x_scale": [1.6, .499]},
                   oracle_assisted=False, physical_truth_evaluated=False, heldout_evaluated=False,
                   provenance=provenance, run_purpose="registered_B_seed101")
        path = directory / "result.json"
        write_json(path, row)
        indexed.append({**row, "result": runner.file_ref(path, source)})
    write_json(source / "run.json", {"test_fixture": True, "provenance": provenance,
                                     "training_fits": {"fits": indexed}})
    return root, source


def mutate_cached_row(source, mutate):
    """Change both mock index and its hashed file, to test semantic checks."""
    run_path = source / "run.json"
    run = json.loads(run_path.read_text())
    indexed = run["training_fits"]["fits"][0]
    path = source / indexed["result"]["path"]
    row = json.loads(path.read_text())
    mutate(row)
    write_json(path, row)
    run["training_fits"]["fits"][0] = {**row, "result": runner.file_ref(path, source)}
    write_json(run_path, run)


def test_cache_loads_actual_hashed_rows(cached_root, training, study):
    root, source = cached_root
    rows, actual_source = runner.cached_B(root, training, study, root / "model/pendulum.xml")
    assert actual_source == source
    assert [row["start_id"] for row in rows] == ["S0", "S1", "S2"]
    assert all(row["input_array_sha256"] == runner.training_hashes(training) for row in rows)


def test_cache_rejects_index_disagreeing_with_hashed_numeric_record(cached_root, training, study):
    root, source = cached_root
    path = source / "run.json"
    run = json.loads(path.read_text())
    run["training_fits"]["fits"][0]["optimized_vector"] = [.5, .4]
    write_json(path, run)
    with pytest.raises(ValueError, match="index|record"):
        runner.cached_B(root, training, study, root / "model/pendulum.xml")


def test_cache_rejects_numeric_file_hash_change(cached_root, training, study):
    root, source = cached_root
    path = source / "fits/B_seed101_S0/result.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed|hash"):
        runner.cached_B(root, training, study, root / "model/pendulum.xml")


@pytest.mark.parametrize("file", ["configs/evidence_v1.json", "model/pendulum.xml"])
def test_cache_rejects_changed_config_or_model(cached_root, training, study, file):
    root, _ = cached_root
    path = root / file
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="config|model"):
        runner.cached_B(root, training, study, root / "model/pendulum.xml")


def test_cache_rejects_changed_training_arrays(cached_root, training, study):
    root, _ = cached_root
    altered = replace(training.sequences[0], measured_qpos=np.ones((5, 1)))
    changed = replace(training, sequences=(altered, training.sequences[1]))
    with pytest.raises(ValueError, match="training|arrays"):
        runner.cached_B(root, changed, study, root / "model/pendulum.xml")


@pytest.mark.parametrize("field,value", [
    ("cell", "E"), ("seed", 102), ("fit_id", "weak_seed101_S0"),
    ("start_id", "S1"), ("free_parameters", list(reversed(FREE))),
    ("starting_values", {"mass_kg": 1.2, "damping_nm_s_per_rad": .15}),
    ("training_ids", ["heldout_chirp", "heldout_pulse"]),
    ("resolved_optimizer_options", {"max_iters": 1}),
    ("optimizer_options", {"backend": "unregistered"}),
])
def test_cache_semantics_cannot_be_changed_by_rehashing(cached_root, training, study, field, value):
    root, source = cached_root
    mutate_cached_row(source, lambda row: row.update({field: value}))
    with pytest.raises(ValueError):
        runner.cached_B(root, training, study, root / "model/pendulum.xml")


def test_cache_rejects_changed_known_nuisance_physics(cached_root, training, study):
    root, source = cached_root
    mutate_cached_row(source, lambda row: row["initial_physics"].update(motor_gain_nm_per_unit=2.))
    with pytest.raises(ValueError):
        runner.cached_B(root, training, study, root / "model/pendulum.xml")


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_cache_requires_complete_registered_start_set(cached_root, training, study, kind):
    root, source = cached_root
    path = source / "run.json"
    run = json.loads(path.read_text())
    rows = run["training_fits"]["fits"]
    if kind == "missing":
        rows.pop()
    else:
        rows[2] = rows[1]
    write_json(path, run)
    with pytest.raises(ValueError):
        runner.cached_B(root, training, study, root / "model/pendulum.xml")


class MockParams:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def as_vector(self):
        return self.values.copy()

    def get_bounds(self):
        return np.array([.4, .001]), np.array([2., .5])


def mocked_problem(*, fail=False):
    problem = SimpleNamespace(optimizer_options={"x_scale": np.array([1.6, .499])},
                              optimization_invoked=False, params=MockParams([.8, .04]),
                              last_trace=[], call_counts={"optimization": {"residual_calls": 1,
                                            "logical_trajectory_rollouts": 2}})

    def run():
        problem.optimization_invoked = True
        problem.last_trace = [{"event": "initial", "objective": 3., "candidate": [.8, .04]}]
        if fail:
            raise RuntimeError("mocked optimizer failure")
        return SimpleNamespace(optimized_params=MockParams([1.2, .15]),
               optimize_result=OptimizeResult(success=False, status=0, message="mock budget exhausted",
                    x=np.array([1.2, .15]), fun=np.array([1., -1.]), jac=np.eye(2), cost=1., nfev=1),
               initial_residual=np.array([2., -2.]), final_residual=np.array([1., -1.]),
               objective_trace=tuple(problem.last_trace), call_counts=problem.call_counts)

    problem.run = run
    return problem


def test_worker_report_failure_keeps_finite_native_candidate(tmp_path, training, study, provenance, monkeypatch):
    monkeypatch.setattr(runner, "build_problem", lambda *args, **kwargs: mocked_problem())
    monkeypatch.setattr(runner, "summarize_optimization", lambda *args, **kwargs: {"test_fixture": True})

    def fail_report(*args, **kwargs):
        raise RuntimeError("mocked report failure")

    monkeypatch.setattr(runner, "save_official_fit", fail_report)
    row = runner.fit_one(training, study, MODEL, study.nominal, FREE,
                         {"mass_kg": .8, "damping_nm_s_per_rad": .04},
                         "B_seed101_S0", "B", "S0", tmp_path / "fit", provenance)
    assert row["status"] == "post_optimization_failed"
    assert row["optimization_invoked"] and row["optimization_returned"]
    assert row["optimizer_success"] is False
    assert row["final_objective"] == 1.
    for name in ("native_result.pkl", "residuals.npz", "parameters.json", "objective_trace.json"):
        assert (tmp_path / "fit" / name).is_file()
    with np.load(tmp_path / "fit/residuals.npz", allow_pickle=False) as arrays:
        assert .5 * arrays["final"] @ arrays["final"] == row["final_objective"]
    assert runner.select_rows([row], FREE)["selected_fit_id"] == row["fit_id"]


def test_worker_optimizer_failure_retains_actual_flag_and_partial_ledger(tmp_path, training, study, provenance, monkeypatch):
    monkeypatch.setattr(runner, "build_problem", lambda *args, **kwargs: mocked_problem(fail=True))
    row = runner.fit_one(training, study, MODEL, study.nominal, FREE,
                         {"mass_kg": .8, "damping_nm_s_per_rad": .04},
                         "B_seed101_S0", "B", "S0", tmp_path / "fit", provenance)
    assert row["status"] == "optimization_failed"
    assert row["optimization_invoked"] is True and row["optimization_returned"] is False
    trace = json.loads((tmp_path / "fit/objective_trace.json").read_text())
    assert [event["event"] for event in trace] == ["initial"]
    assert row["call_counts"]["optimization"]["residual_calls"] == 1
    assert row["call_counts"]["optimization"]["logical_trajectory_rollouts"] == 2
    assert runner.select_rows([row], FREE)["selected_fit_id"] is None


def test_worker_interruption_preserves_flag_partial_trace_and_ledger(tmp_path, training, study, provenance, monkeypatch):
    problem = mocked_problem()

    def interrupted():
        problem.optimization_invoked = True
        problem.last_trace = [{"event": "initial", "candidate": [.8, .04], "objective": 3.}]
        raise KeyboardInterrupt()

    problem.run = interrupted
    monkeypatch.setattr(runner, "build_problem", lambda *args, **kwargs: problem)
    with pytest.raises(KeyboardInterrupt):
        runner.fit_one(training, study, MODEL, study.nominal, FREE,
                       {"mass_kg": .8, "damping_nm_s_per_rad": .04},
                       "B_seed101_S0", "B", "S0", tmp_path / "fit", provenance)
    row = json.loads((tmp_path / "fit/result.json").read_text())
    assert row["status"] == "interrupted"
    assert row["optimization_invoked"] and not row["optimization_returned"]
    assert row["call_counts"]["optimization"]["residual_calls"] == 1
    assert (tmp_path / "fit/objective_trace.json").is_file()


def test_worker_prefit_failure_does_not_claim_optimizer_invocation(tmp_path, training, study, provenance, monkeypatch):
    def invalid_problem(*args, **kwargs):
        raise ValueError("mocked input validation failure")

    monkeypatch.setattr(runner, "build_problem", invalid_problem)
    row = runner.fit_one(training, study, MODEL, study.nominal, FREE,
                         {"mass_kg": .8, "damping_nm_s_per_rad": .04},
                         "B_seed101_S0", "B", "S0", tmp_path / "fit", provenance)
    assert row["optimization_invoked"] is False and row["optimization_returned"] is False
    assert row["failure"]["type"] == "ValueError"


def test_worker_cannot_overwrite_existing_fit(tmp_path, training, study, provenance, monkeypatch):
    directory = tmp_path / "fit"
    directory.mkdir()
    (directory / "result.json").write_text("retained test fixture")
    with pytest.raises(FileExistsError):
        runner.fit_one(training, study, MODEL, study.nominal, FREE,
                       {"mass_kg": .8, "damping_nm_s_per_rad": .04},
                       "B_seed101_S0", "B", "S0", directory, provenance)
    assert (directory / "result.json").read_text() == "retained test fixture"


@pytest.fixture
def mocked_study(tmp_path, training, study, provenance, monkeypatch):
    """Exercise the real orchestration on fake, clearly labeled worker outputs."""
    from mujoco_sysid_lab import advanced_data, api_check, full_evaluation, joint_experiment

    root = tmp_path / "mock_matrix"
    (root / "configs").mkdir(parents=True)
    (root / "model").mkdir()
    for name in ("evidence_v1.json", "ground_truth_v1.json"):
        (root / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    (root / "model/pendulum.xml").write_bytes(MODEL.read_bytes())
    output = root / "artifacts/mock_test_matrix"
    args = SimpleNamespace(config="configs/evidence_v1.json", registration="mock_registration.json")
    state = {"worker_calls": [], "cache_calls": [], "evaluations": [],
             "fail_B_physics": False, "fail_delay_accuracy": False}
    def collect(*args, **kwargs):
        state["provenance_command"] = list(args[3])
        return dict(provenance)

    monkeypatch.setattr(runner, "collect_provenance", collect)
    monkeypatch.setattr(runner, "snapshot_sources", lambda *args, **kwargs: {"test_fixture": True})
    monkeypatch.setattr(joint_experiment, "check_registration", lambda *args, **kwargs: {"test_fixture": True})
    monkeypatch.setattr(api_check, "check_api", lambda *args, **kwargs: {"test_fixture": True})
    monkeypatch.setattr(runner, "cached_B", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "cached_units", lambda *args, **kwargs: None)

    def cache6(root, cell, data, *args, **kwargs):
        state["cache_calls"].append((cell, data.seed))
        if (state["fail_B_physics"] and cell in ("C_delay_only", "C_joint")) or (
                state["fail_delay_accuracy"] and cell == "C_joint"):
            raise AssertionError("Cache lookup bypassed a failed prerequisite")
        return None

    monkeypatch.setattr(runner, "cached_milestone6", cache6)

    def create(study, truth, model_path, seed, scenario):
        if scenario == "weak":
            sequence = replace(training.sequences[0], trajectory_name="weak_step", trajectory_id=4)
            safe = TrainingDataset(seed, (sequence,))
        else:
            safe = replace(training, seed=seed)
        return SimpleNamespace(training=safe, seed=seed, scenario=scenario,
                               delay_s=.024 if scenario == "delayed" else 0.)

    def write(output, data, provenance):
        output.mkdir(parents=True, exist_ok=False)
        manifest = output / "manifest.json"
        write_json(manifest, {"test_fixture": True, "seed": data.seed, "scenario": data.scenario})
        return manifest

    monkeypatch.setattr(advanced_data, "create_scenario", create)
    monkeypatch.setattr(advanced_data, "write_scenario", write)

    def worker(data, study, model_path, physical, free, start, fit_id, cell, start_id, directory, provenance):
        journal = json.loads((output / "run.json").read_text())
        prepared = next(row for row in journal["fits"] if row["fit_id"] == fit_id)
        assert prepared["status"] == "prepared"
        assert journal["progress"]["active_fit"] == fit_id
        assert isinstance(data, TrainingDataset)
        assert not hasattr(data, "records") and not hasattr(data, "metadata")
        assert all(sequence.trajectory_name.startswith("train_") or sequence.trajectory_name == "weak_step"
                   for sequence in data.sequences)
        state["worker_calls"].append({"cell": cell, "seed": data.seed, "start_id": start_id,
                                     "free": free, "start": dict(start), "physical": physical,
                                     "training_ids": tuple(item.trajectory_name for item in data.sequences)})
        values = {"mass_kg": 1.2, "damping_nm_s_per_rad": .15, "delay_s": .024}
        if cell == "B" and state["fail_B_physics"]:
            values.update(mass_kg=.8, damping_nm_s_per_rad=.04)
        if cell == "C_delay_only" and state["fail_delay_accuracy"]:
            values["delay_s"] = .04
        row = candidate(start_id, float(len(data.sequences)), cell=cell, seed=data.seed,
                        optimized_vector=[values[name] for name in free], free_parameters=list(free))
        row.update(initial_physics=config_to_dict(physical), starting_values=dict(start),
                   input_array_sha256=runner.training_hashes(data), provenance=provenance,
                   optimization_summary={"test_fixture": True, "physical_truth_evaluated": False,
                     "parameter_rows": [{"name": name, "estimate": values[name]} for name in free]})
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "result.json", row)
        return row

    monkeypatch.setattr(runner, "fit_one", worker)

    def evaluate(data, physical, model_path, study, *, delay_s=0.):
        # Current group exists before evaluation, so any missing seal indicates
        # evaluation preceded training-only selection.
        assert all((directory / "selection.json").is_file() for directory in (output / "groups").iterdir())
        journal = json.loads((output / "run.json").read_text())
        assert journal["progress"]["phase"] == "evaluation"
        assert all(row["status"] in ("completed", "not_run") for row in journal["fits"])
        state["evaluations"].append((data.scenario, data.seed, physical, delay_s))
        train_rows = [{"trajectory_id": sequence.trajectory_name, "n_samples": 1,
                       "normalized_rmse": 1.} for sequence in data.training.sequences]
        train = {"qpos_rmse": 1., "qvel_rmse": 1., "normalized_rmse": 1.,
                 "trajectory_metrics": train_rows}
        return {"train": train, "heldout": {**train, "trajectory_metrics": []},
                "generator_aware_noise_floor": {"test_fixture": True}}

    monkeypatch.setattr(full_evaluation, "evaluate_model", evaluate)
    monkeypatch.setattr(full_evaluation, "phase_lag_diagnostics",
                        lambda *args, **kwargs: {"test_fixture": True, "delay_s": kwargs["delay_s"]})
    return root, output, args, state


def test_full_matrix_accounts_for_all_82_fits_and_five_seeds(mocked_study):
    root, output, args, state = mocked_study
    assert runner.run_study(root, args, output, through=9) == 0
    result = json.loads((output / "run.json").read_text())
    assert result["status"] == "passed"
    assert len(result["fits"]) == result["registered_matrix_total"] == 82
    assert result["optimization_calls_this_invocation"] == 82
    assert result["reused_registered_fits"] == 0
    expected = {"A1": 1, "A2": 1, "B": 15, "weak": 15, "delay_excluded": 15,
                "C_delay_only": 5, "C_joint": 15, "E": 15}
    for cell, count in expected.items():
        rows = [row for row in result["fits"] if row["cell"] == cell]
        assert len(rows) == count
        assert {row["seed"] for row in rows} == ({101} if cell.startswith("A") else set(range(101, 106)))
    assert len({row["fit_id"] for row in result["fits"]}) == 82
    assert len(result["groups"]) == 30
    for group in result["groups"]:
        seal = root / group["selection"]["path"]
        assert group["selection_unchanged"] is True
        assert sha256_file(seal) == group["selection"]["sha256"]
        selected = json.loads(seal.read_text())
        assert selected["selected_fit_id"] == f"{group['group_id']}_S0"
        assert selected["heldout_evaluated"] is False
        assert selected["physical_truth_evaluated"] is False
    assert state["evaluations"]
    for call in state["worker_calls"]:
        if call["cell"] == "weak":
            assert call["training_ids"] == ("weak_step",)
        if call["cell"] == "C_delay_only":
            assert call["physical"].mass_kg == 1.2 and call["physical"].damping_nm_s_per_rad == .15
            assert call["start"] == {"delay_s": .002}


@pytest.mark.parametrize("within_checkout", [False, True])
def test_study_command_records_the_actual_interpreter(mocked_study, monkeypatch, within_checkout):
    root, output, args, state = mocked_study
    executable = root/".venv/bin/python" if within_checkout else root.parent/"external-python/bin/python"
    # Fake workers keep this a command/provenance regression, with no child
    # process or scientific optimization launched using the labeled path.
    monkeypatch.setattr(runner.sys, "executable", str(executable))
    assert runner.run_study(root, args, output, through=6) == 0
    assert state["provenance_command"][0] == (".venv/bin/python" if within_checkout else str(executable))
    assert state["provenance_command"][1] == "scripts/run_experiment.py"


@pytest.mark.parametrize("failure,expected_not_run", [("fail_B_physics", 20), ("fail_delay_accuracy", 15)])
def test_failed_prerequisites_retain_not_run_rows_and_cannot_use_cache(mocked_study, failure, expected_not_run):
    root, output, args, state = mocked_study
    state[failure] = True
    assert runner.run_study(root, args, output, through=9) == 1
    result = json.loads((output / "run.json").read_text())
    assert result["status"] == "incomplete"
    assert len(result["fits"]) == 82
    not_run = [row for row in result["fits"] if row["status"] == "not_run"]
    assert len(not_run) == expected_not_run
    assert all(row["optimization_invoked"] is False and row["optimized_vector"] is None for row in not_run)
    assert result["optimization_calls_this_invocation"] == 82 - expected_not_run
    cells = {row["cell"] for row in not_run}
    assert cells == ({"C_delay_only", "C_joint"} if failure == "fail_B_physics" else {"C_joint"})
    assert all(row["status"] == "incomplete" for row in result["milestones"])


def test_milestone6_only_runs_seed101_registered_delay_cells(mocked_study):
    root, output, args, state = mocked_study
    assert runner.run_study(root, args, output, through=6) == 0
    result = json.loads((output / "run.json").read_text())
    assert len(result["fits"]) == 10
    assert {row["cell"] for row in result["fits"]} == {"B", "delay_excluded", "C_delay_only", "C_joint"}
    assert {row["seed"] for row in result["fits"]} == {101}
    assert result["registered_matrix_total"] == 82
    assert result["through_milestone"] == 6


def test_fresh_disables_every_cache_lookup(mocked_study, monkeypatch):
    root, output, args, _ = mocked_study
    args.fresh = True

    def forbidden(*args, **kwargs):
        raise AssertionError("Fresh execution consulted a cache")

    for name in ("cached_units", "cached_B", "cached_milestone6"):
        monkeypatch.setattr(runner, name, forbidden)
    assert runner.run_study(root, args, output, through=9) == 0
    result = json.loads((output / "run.json").read_text())
    assert result["cache_policy"]["fresh"]
    assert result["optimization_calls_this_invocation"] == 82
    assert result["reused_registered_fits"] == 0


def test_cache_override_flags_are_passed_to_exact_worker_cache(mocked_study, monkeypatch):
    root, output, args, _ = mocked_study
    args.unit_cache_run = "artifacts/custom_units"
    args.physical_cache_run = "artifacts/custom_physical"
    args.delay_cache_run = "artifacts/custom_delay"
    calls = []

    def cache(kind):
        def call(*args, cache_run):
            calls.append((kind, cache_run))
            return None
        return call

    monkeypatch.setattr(runner, "cached_units", cache("unit"))
    monkeypatch.setattr(runner, "cached_B", cache("physical"))
    monkeypatch.setattr(runner, "cached_milestone6", cache("delay"))
    assert runner.run_study(root, args, output, through=9) == 0
    assert set(calls) == {("unit", args.unit_cache_run), ("physical", args.physical_cache_run), ("delay", args.delay_cache_run)}


@pytest.mark.parametrize("error", [RuntimeError("mock evaluation failure"), KeyboardInterrupt()])
def test_progress_journal_contains_completed_numerical_rows_before_evaluation_failure(mocked_study, monkeypatch, error):
    from mujoco_sysid_lab import full_evaluation
    root, output, args, _ = mocked_study

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(full_evaluation, "evaluate_model", failed)
    with pytest.raises(type(error)):
        runner.run_study(root, args, output, through=6)
    journal = json.loads((output / "run.json").read_text())
    assert journal["progress"]["active_group"] == "B_seed101"
    assert journal["progress"]["phase"] == "evaluation"
    assert len(journal["fits"]) == 3
    assert all(row["status"] == "completed" and row["optimization_returned"] for row in journal["fits"])
    assert (output / "groups/B_seed101/selection.json").is_file()
    for row in journal["fits"]:
        assert sha256_file(root / row["source_result"]["path"]) == row["source_result"]["sha256"]


def test_dataset_generation_failure_has_an_initial_progress_journal(mocked_study, monkeypatch):
    from mujoco_sysid_lab import advanced_data
    root, output, args, _ = mocked_study

    def failed(*args, **kwargs):
        raise RuntimeError("mock dataset failure")

    monkeypatch.setattr(advanced_data, "create_scenario", failed)
    with pytest.raises(RuntimeError, match="dataset failure"):
        runner.run_study(root, args, output, through=6)
    journal = json.loads((output / "run.json").read_text())
    assert journal["fits"] == []
    assert journal["progress"]["phase"] == "dataset_generation"
    assert journal["progress"]["active_dataset"] == "matched_seed101"


def test_cache_directory_optional_default_explicit_override_and_scope(tmp_path):
    assert runner.cache_directory(tmp_path, None, "unit") == tmp_path / runner.LEGACY_CACHE_RUNS["unit"]
    custom = tmp_path / "artifacts/custom"
    custom.mkdir(parents=True)
    (custom / "run.json").write_text('{"test_fixture":true}')
    assert runner.cache_directory(tmp_path, "artifacts/custom/run.json", "unit") == custom
    with pytest.raises(ValueError, match="missing"):
        runner.cache_directory(tmp_path, "artifacts/missing", "unit")
    with pytest.raises(ValueError, match="artifacts"):
        runner.cache_directory(tmp_path, "../outside", "unit")


@pytest.fixture
def cached_source_tree(tmp_path):
    # Local read-only source clone; no evidence runs and no project commits.
    root = tmp_path / "source_fixture"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(ROOT), str(root)], check=True)
    return root


def test_cache_source_verifies_original_git_hashes_and_permits_legacy_code(cached_source_tree):
    root = cached_source_tree
    record = collect_provenance(root, root / "configs/evidence_v1.json", root / "model/pendulum.xml",
                                ["python", "-m", "pytest", "labeled source fixture"])
    assert record["git"]["dirty"] is False
    (root / "src/mujoco_sysid_lab/api_check.py").write_text("# later checkout change\n")
    result = runner.verify_cache_source(root, record)
    assert result["source_dirty"] is False
    assert result["current_source_byte_equality_required"] is False
    original_hash = record["source_sha256"]["src/mujoco_sysid_lab/api_check.py"]
    record["source_sha256"]["src/mujoco_sysid_lab/api_check.py"] = "0" * 64
    with pytest.raises(ValueError, match="source hash changed"):
        runner.verify_cache_source(root, record)
    record["source_sha256"]["src/mujoco_sysid_lab/api_check.py"] = original_hash
    record["package_versions"]["numpy"] = "unregistered fixture version"
    with pytest.raises(ValueError, match="libraries differ"):
        runner.verify_cache_source(root, record)


def test_cache_source_dirty_patch_reconstruction_and_tamper_rejection(cached_source_tree):
    from mujoco_sysid_lab.provenance import snapshot_sources
    root = cached_source_tree
    path = root / "src/mujoco_sysid_lab/api_check.py"
    path.write_text(path.read_text() + "\n# authored dirty source fixture\n")
    record = collect_provenance(root, root / "configs/evidence_v1.json", root / "model/pendulum.xml",
                                ["python", "-m", "pytest", "labeled dirty source fixture"])
    assert record["git"]["dirty"] is True
    with pytest.raises(ValueError, match="snapshot"):
        runner.verify_cache_source(root, record)
    record["source_snapshot"] = snapshot_sources(root, root / "artifacts/source_snapshot")
    assert runner.verify_cache_source(root, record)["source_dirty"] is True
    patch = root / "artifacts/source_snapshot/tracked.patch"
    patch.write_text(patch.read_text() + "\n")
    with pytest.raises(ValueError, match="ledger changed"):
        runner.verify_cache_source(root, record)
