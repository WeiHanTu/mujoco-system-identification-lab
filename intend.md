# Intent and claim boundaries

Project: `mujoco-system-identification-lab`.
Status: Milestones 1–3 are verified. Deterministic matched-model data, nominal/oracle baselines, and independent oracle-assisted mass/damping recovery checks exist. Joint recovery, fitted held-out improvement, and multi-seed robustness remain unverified.

## Intended result

Build a small, inspectable study showing the full estimation, validation, and diagnostic workflow: generate deterministic synthetic observations from hidden dynamics; initialize a biased model; estimate selected parameters with MuJoCo's official SysID toolbox; evaluate unseen complete trajectories; and test excitation, timing, initialization, noise, and model mismatch.

The primary result is held-out prediction improvement. Parameter recovery matters because a good trajectory fit can conceal incorrect physical interpretation. Training loss is an optimization diagnostic, not the headline result.

Approved description, usable only with the actual achieved scope:

> A reproducible synthetic dynamics system-identification study that exercises the estimation, validation, and diagnostic workflow needed before applying the method to real traces.

## Scope

One actuated hinge and a rigid cylindrical link with fixed geometry. Estimate link mass and joint viscous damping. Later estimate one shared positive sensor delay for position and velocity. Keep motor torque gain known. Use deterministic measurement noise and complete-trajectory splits.

The small system makes physical assumptions, parameter sensitivities, and implementation mistakes visible. It does not establish that arbitrary robot parameters are identifiable. Fixing geometry and gain is part of the experiment's contract, not a capability demonstrated by the estimator.

The stress test adds a smooth Coulomb-like joint torque to the target generator only, while fitting mass and viscous damping with the simpler model. Preserve a negative or inconclusive result if that perturbation does not produce substantial bias.

## Public claims after verification

| Claim | Evidence required |
|---|---|
| Synthetic mass/damping recovery | All registered seed/start rows, errors against truth, bound/convergence diagnostics |
| Held-out synthetic prediction | Chirp and pulse results from controls excluded from estimation and selection |
| Excitation sensitivity | Time/energy-accounted step versus rich-excitation comparison, without requiring the step to fail |
| Timing sensitivity | Zero-delay and delay-aware comparisons, delay error, phase-lag diagnostic, safe shared windows |
| Initialization/noise sensitivity | Five seeds and three starts, all failures retained, median/min/max |
| Model-mismatch limitations | Residual improvement, physical bias, sensitivity/fit diagnostics, both oracle definitions |
| Practical CPU execution | Measured runtime and exact CPU/software/thread environment |

Until that evidence exists, describe these as planned experiments. Passing a smoke test is not full-study verification.

## Not demonstrated

- Real robot data or real-robot validation.
- Hardware-in-the-loop.
- Physical actuator calibration.
- Contact-rich manipulation.
- Real sim-to-real transfer.
- Production deployment.
- General identifiability for arbitrary robots.

The experiment uses the same simulator family for generation and fitting. This controlled setting is useful for checking a workflow and makes the matched-model task easier than real data. Adding one omitted effect does not close that gap.

## Interview material and privacy

At completion, write `private/interview_notes.md`, ignored by Git before creation. Include the user's 30-second English and Traditional Chinese scripts, adjusted only if necessary to match what actually passed. Do not use completed-tense claims for failed or unfinished stages.

Provide short bilingual follow-ups on: choosing 1-DOF; multiple excitation trajectories; splitting by trajectory; mass/damping/gain/delay confounding; biased physics despite better prediction; real-trace ingestion; checks before applying fitted parameters; CPU versus MJX; scaling to manipulators/contacts; and what HIL would add. Explain assumptions and limitations directly, without presenting this synthetic project as hardware experience.

## Completion definition

All requested public artifacts, tests, a short CI experiment, the separately documented full evidence command, official HTML reporting, archived raw synthetic evidence, and ignored interview notes exist. Every result is traceable to configuration and source provenance. Success criteria are reported as pass/fail/inconclusive, with reasons. A scientifically negative result can still make a complete project; fabricated or untraceable success cannot.
