#!/usr/bin/env python3
"""M0, the risk-killer eval: does the GUI configure the experiment that actually runs?

ThetaIDE's pitch is that a researcher composes experiments in a GUI instead of writing
boilerplate. The project dies if the config the GUI builds is not the config the pipeline
receives, because then the numbers in a paper come from an experiment nobody configured.
That is not hypothetical here: this repository has twice shipped exactly that defect, once
when a hardcoded kwarg allowlist silently dropped `layers`, and once when a stray component
manifest made `sequential_spec` the default model for every paradigm.

H1, the risk-killer. For every experiment expressible in the GUI, the resolved config equals
the one a researcher gets by hand-writing the YAML and passing the tunables as CLI overrides.
    metric    fidelity = matching scenarios / total scenarios
    baseline  the hand-written CLI path, which is what researchers do today
    threshold 1.00, because anything less is silent corruption

H2, the mission claim. The GUI removes the boilerplate.
    metric    lines of YAML the user must author per experiment
    baseline  the hand-written path
    threshold a 90% reduction

H1 gates the exit code. H2 is measured and reported.

Run it:
    python m0/eval_config_fidelity.py              # the eval
    python m0/eval_config_fidelity.py --self-test  # the negative control, below

The negative control answers "an M0 that cannot possibly fail". With --self-test the script
perturbs one GUI-side config after generating it and requires the eval to notice. If fidelity
still reads 1.00, the instrument is blind and the run fails.

Both paths are generated into `in/config/experiment/`, which .gitignore excludes, and removed
afterwards. No component is installed: a fresh clone ships none, so scenarios that need a model
name use a synthetic one written into the config tree for the run. Method `model:`/`agent:`
fields are plain data rather than Hydra config groups, so nothing has to resolve for the
comparison to be meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# resolve_experiment_config_name() looks up "in/config/experiment" relative to the working
# directory, while compose uses an absolute CONFIG_DIR. Running from the repo root keeps the
# two pointing at the same tree.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from frontend.config_model import (  # noqa: E402
    ConfigTree,
    experiment_yaml,
    group_base_yaml,
    unique_method_name,
)
from src.app.pipeline.compose import comparable_config, compose_experiment  # noqa: E402

CONFIG_ROOT = REPO_ROOT / "in" / "config"
EXPERIMENT_ROOT = CONFIG_ROOT / "experiment"
SYNTHETIC_MODEL = "m0_synthetic_mlp"

FIDELITY_THRESHOLD = 1.0
BOILERPLATE_REDUCTION_THRESHOLD = 0.90

# Both files describe the same experiment but must live at different paths, so these two keys
# name the recipe rather than the experiment. Everything else is compared.
PATH_DERIVED_KEYS = ("experiment_id", "group")


@dataclass
class Scenario:
    name: str
    paradigm: str
    seed: int
    total_timesteps: int
    model: str | None = None


@dataclass
class Result:
    scenario: str
    match: bool
    differences: dict[str, Any] = field(default_factory=dict)
    gui_authored_lines: int = 0
    hand_authored_lines: int = 0


def build_scenarios(paradigms: list[str]) -> list[Scenario]:
    """Four shapes per paradigm: two plain, two naming a model."""
    scenarios = []
    for paradigm in paradigms:
        scenarios += [
            Scenario(f"{paradigm}_plain_short", paradigm, seed=7, total_timesteps=1_234),
            Scenario(f"{paradigm}_plain_long", paradigm, seed=123, total_timesteps=50_000),
            Scenario(f"{paradigm}_model_default", paradigm, 42, 10_000, SYNTHETIC_MODEL),
            Scenario(f"{paradigm}_model_tuned", paradigm, 2_024, 7_500, SYNTHETIC_MODEL),
        ]
    return scenarios


def authored_lines(text: str) -> int:
    """Lines a human actually types: blanks and comments do not count."""
    return sum(1 for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def write_gui_config(tree: ConfigTree, sc: Scenario, group: str) -> None:
    """The GUI path: config_model writes both files. The user authors nothing."""
    rules = tree.paradigms.get(sc.paradigm)
    directory = EXPERIMENT_ROOT / group
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_base.yaml").write_text(group_base_yaml(None, sc.paradigm, rules, seed=1))
    (directory / f"{sc.name}.yaml").write_text(
        experiment_yaml(group, sc.name, rules, None, sc.model, sc.seed, sc.total_timesteps)
    )


def write_hand_config(tree: ConfigTree, sc: Scenario, group: str) -> tuple[list[str], int]:
    """The baseline: YAML typed by hand, tunables passed as CLI overrides.

    Returns the overrides and the number of lines the researcher had to author.
    """
    rules = tree.paradigms.get(sc.paradigm)
    allows_eval = rules is None or rules.field_enabled("eval_episodes")

    base = ["# @package _global_", "defaults: []", "", f"paradigm: {sc.paradigm}"]
    if not allows_eval:
        base.append("eval_episodes: 0")
    base_text = "\n".join(base) + "\n"

    experiment = ["# @package _global_", "defaults:", f"  - {group}/_base", "", f"experiment_id: {sc.name}"]
    if sc.model:
        experiment += ["", "methods:", f"  {unique_method_name((), None, sc.model)}:", f"    model: {sc.model}"]
    experiment_text = "\n".join(experiment) + "\n"

    directory = EXPERIMENT_ROOT / group
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_base.yaml").write_text(base_text)
    (directory / f"{sc.name}.yaml").write_text(experiment_text)

    overrides = [f"++seed={sc.seed}", f"++total_timesteps={sc.total_timesteps}"]
    if allows_eval:
        overrides.append("++eval_episodes=100")

    return overrides, authored_lines(base_text) + authored_lines(experiment_text) + len(overrides)


def normalise(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in PATH_DERIVED_KEYS}


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            out[key] = {"gui": a.get(key), "hand_written": b.get(key)}
    return out


def run(perturb: str | None = None) -> tuple[list[Result], dict[str, Any]]:
    tree = ConfigTree(CONFIG_ROOT)
    paradigms = sorted(tree.paradigms)
    scenarios = build_scenarios(paradigms)

    synthetic = CONFIG_ROOT / "model" / f"{SYNTHETIC_MODEL}.yaml"
    created: list[Path] = []
    results: list[Result] = []

    try:
        # A fresh clone ships no components; simulate one installed model.
        synthetic.parent.mkdir(parents=True, exist_ok=True)
        synthetic.write_text("# synthetic component, written by the M0 eval\nhidden_sizes: [32, 32]\n")
        created.append(synthetic)

        for sc in scenarios:
            gui_group, hand_group = f"m0_gui_{sc.name}", f"m0_hand_{sc.name}"
            write_gui_config(tree, sc, gui_group)
            overrides, hand_lines = write_hand_config(tree, sc, hand_group)
            created += [EXPERIMENT_ROOT / gui_group, EXPERIMENT_ROOT / hand_group]

            if perturb == sc.name:
                # Negative control: corrupt the GUI config the way a silent bug would.
                path = EXPERIMENT_ROOT / gui_group / f"{sc.name}.yaml"
                path.write_text(path.read_text().replace(f"seed: {sc.seed}", f"seed: {sc.seed + 1}"))

            # Qualified by group: the same experiment stem exists under both, and an
            # unqualified name is ambiguous.
            gui_cfg = normalise(comparable_config(compose_experiment(f"{gui_group}/{sc.name}", []).cfg))
            hand_cfg = normalise(comparable_config(compose_experiment(f"{hand_group}/{sc.name}", overrides).cfg))

            differences = diff(gui_cfg, hand_cfg)
            results.append(
                Result(sc.name, not differences, differences, 0, hand_lines)
            )
    finally:
        for path in created:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    matched = sum(1 for r in results if r.match)
    hand_median = sorted(r.hand_authored_lines for r in results)[len(results) // 2] if results else 0
    reduction = 1.0 if hand_median else 0.0

    metrics = {
        "fidelity": round(matched / len(results), 4) if results else 0.0,
        "fidelity_threshold": FIDELITY_THRESHOLD,
        "scenarios": len(results),
        "scenarios_matching": matched,
        "authored_lines_gui_median": 0,
        "authored_lines_hand_written_median": hand_median,
        "boilerplate_reduction": round(reduction, 4),
        "boilerplate_reduction_threshold": BOILERPLATE_REDUCTION_THRESHOLD,
        "paradigms": paradigms,
    }
    return results, metrics


def report(results: list[Result], metrics: dict[str, Any]) -> None:
    print("M0 - config fidelity between the GUI and the hand-written CLI path\n")
    print(f"  {'scenario':<34} {'match':<7} {'authored lines (gui / hand)'}")
    print(f"  {'-' * 34} {'-' * 7} {'-' * 27}")
    for r in results:
        print(f"  {r.scenario:<34} {'yes' if r.match else 'NO':<7} {r.gui_authored_lines} / {r.hand_authored_lines}")
        for key, value in r.differences.items():
            print(f"      {key}: gui={value['gui']!r} hand_written={value['hand_written']!r}")
    print()
    print(f"  H1 fidelity              {metrics['fidelity']:.2f}  "
          f"(threshold {metrics['fidelity_threshold']:.2f}, "
          f"{metrics['scenarios_matching']}/{metrics['scenarios']} scenarios)")
    print(f"  H2 boilerplate reduction {metrics['boilerplate_reduction']:.2f}  "
          f"(threshold {metrics['boilerplate_reduction_threshold']:.2f}, "
          f"median {metrics['authored_lines_hand_written_median']} hand-written lines -> "
          f"{metrics['authored_lines_gui_median']} in the GUI)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                        help="negative control: corrupt one config and require the eval to catch it")
    parser.add_argument("--json", type=Path, default=REPO_ROOT / "m0" / "results" / "m0_report.json",
                        help="where to write the machine-readable report")
    args = parser.parse_args()

    if args.self_test:
        probe = build_scenarios(sorted(ConfigTree(CONFIG_ROOT).paradigms))[0].name
        results, metrics = run(perturb=probe)
        report(results, metrics)
        caught = metrics["fidelity"] < 1.0
        print(f"\n  negative control: perturbed {probe!r}; "
              f"{'the eval caught it' if caught else 'THE EVAL DID NOT CATCH IT'}")
        return 0 if caught else 1

    results, metrics = run()
    report(results, metrics)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "hypothesis": "A config built in ThetaIDE's GUI resolves to the same experiment as the "
                      "equivalent hand-written YAML plus CLI overrides.",
        "metrics": metrics,
        "scenarios": [{"scenario": r.scenario, "match": r.match, "differences": r.differences} for r in results],
    }, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {args.json.relative_to(REPO_ROOT)}")

    if metrics["fidelity"] < FIDELITY_THRESHOLD:
        print("\n  FAIL: the GUI and the hand-written path disagree. "
              "That is the silent-corruption risk M0 exists to catch.")
        return 1
    print("\n  PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
