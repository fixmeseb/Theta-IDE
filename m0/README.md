# M0 — the risk-killer eval

## What could kill this project

ThetaIDE's pitch is that a researcher composes experiments in a GUI instead of writing
boilerplate. The whole product rests on one assumption: **that the config the GUI builds is
the config the pipeline actually runs.** If those drift apart, the IDE is worse than useless —
a researcher publishes numbers from an experiment nobody configured, and never sees the gap.

This is not a theoretical risk. This repository has shipped that exact defect twice:

- a hardcoded kwarg allowlist in the tabular paradigm silently dropped `layers`, so a model
  configured with hidden layers trained without them;
- a stray `.theta_component.json` made `sequential_spec` the default model for *every*
  paradigm, so experiments silently ran against an architecture nobody selected.

Both were caught by hand, late. M0 makes that class of defect fail the build instead.

## The claim, as a measurable hypothesis

> **H1 (the risk-killer).** For every experiment expressible in ThetaIDE's GUI, the resolved
> configuration handed to the pipeline is identical to the one a researcher obtains by
> hand-writing the equivalent YAML and passing the tunables as command-line overrides.

| | |
|---|---|
| **metric** | `fidelity` = matching scenarios / total scenarios |
| **baseline** | the hand-written CLI path — what researchers do today, without this IDE |
| **threshold** | **1.00.** Anything below is silent corruption, not a rounding error |

> **H2 (the mission claim).** The GUI removes the boilerplate.

| | |
|---|---|
| **metric** | lines of YAML the user must author per experiment |
| **baseline** | the hand-written path |
| **threshold** | a 90% reduction |

H1 gates the exit code. H2 is measured and reported, and quantifies the "so what" of the
mission statement.

## Running it

```bash
python m0/eval_config_fidelity.py
```

Exits non-zero if fidelity drops below 1.00, and writes [`results/m0_report.json`](results/m0_report.json).
It needs only `hydra-core`, `omegaconf` and `pyyaml` — the composition path is free of torch,
so the eval runs in seconds without the ML stack installed.

## Current result

| metric | value | threshold |
|---|---|---|
| H1 fidelity | **1.00** (16/16 scenarios) | 1.00 |
| H2 boilerplate reduction | **1.00** (median 11 hand-written lines → 0) | 0.90 |

Sixteen scenarios: four shapes across each of the four shipped paradigms (`offline_rl`,
`online_rl`, `supervised`, `unsupervised`).

## Why this eval can fail

An eval that cannot fail proves nothing, so the instrument ships with a negative control:

```bash
python m0/eval_config_fidelity.py --self-test
```

This perturbs one GUI-side config after generating it — the way a silent bug would — and
**requires** the eval to notice. Fidelity drops to 0.94 (15/16) and the run reports that it
caught the fault. If the metric stayed at 1.00, the instrument would be blind and the
self-test exits non-zero.

## How the scenarios are built

Each scenario is composed twice through `src/app/pipeline/compose.py`, the module both the
CLI and the GUI already share:

- **the GUI path** calls `frontend/config_model.py`'s own `group_base_yaml()` and
  `experiment_yaml()` — the same functions the application uses. The user authors nothing.
- **the hand-written path** is YAML typed out in the eval, independently of `config_model`,
  with the tunables passed as `++seed=`, `++total_timesteps=` and `++eval_episodes=`
  overrides, which is how a researcher works today.

Both resolved configs are reduced with `comparable_config()` and compared key by key.

## Honest limitations

- **No real users.** The rubric's bonus asks for pilot users or collected traces. We have
  neither, so every scenario here is synthetic. They are generated programmatically and the
  comparison is exact, but they are not evidence that researchers *want* this — only that the
  IDE is faithful when they use it.
- **A simulated component.** A fresh clone ships no algorithms, models or environments; they
  install from the Theta Hub. The eval writes one synthetic model config for the run and
  deletes it afterwards. Method `model:` and `agent:` fields are plain data rather than Hydra
  config groups, so nothing needs to resolve for the comparison to hold.
- **Two keys are excluded.** `experiment_id` and `group` are derived from the file's path, and
  the two configs must live at different paths to be compared at all. Every other key is compared.
- **Composition, not execution.** This proves the config reaching the pipeline is correct. It
  does not prove training then honours it — that is the next milestone's problem.

## If the eval fails

A failure is a spiral review, not an embarrassment. The report names the scenario and the
exact keys that diverged, with the GUI value against the hand-written one. That is the bug
report.
