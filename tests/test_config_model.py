"""Tests for the ThetaIDE config model.

These run against the real in/config tree rather than fixtures, so they fail if
the backend's config schema moves in a way the panel would not survive. Where a
concrete value would rot as configs are added, the assertion is written as a
property instead.
"""

import pytest

from frontend.config_model import Agent, ConfigTree, Environment, Paradigm, Selection


@pytest.fixture(scope="module")
def tree():
    return ConfigTree.discover()


# --------------------------------------------------------------- loading


def test_discover_finds_config_tree(tree):
    assert tree.config_root.is_dir()
    assert tree.config_root.name == "config"


def test_known_paradigms_load(tree):
    assert {"online_rl", "offline_rl", "supervised"} <= set(tree.paradigms)


def test_known_agents_load(tree):
    assert {"ppo", "cql", "iql"} <= set(tree.agents)


def test_agents_expose_algorithm_and_hyperparameters(tree):
    ppo = tree.agents["ppo"]
    assert ppo.algorithm == "ppo"
    assert "lr" in ppo.hyperparameters
    assert "algorithm" not in ppo.hyperparameters


def test_models_expose_architecture(tree):
    assert "dnn" in tree.models
    assert "cew" in tree.models
    assert tree.models["dnn"].architecture == "dnn"
    assert "architecture" not in tree.models["dnn"].parameters


def test_environments_carry_offline_only_flag(tree):
    assert tree.environments["cartpole"].offline_only is False
    assert tree.environments["mimic"].offline_only is True


def test_experiments_load_with_groups(tree):
    assert tree.experiments
    assert all(e.group for e in tree.experiments.values())
    assert "cartpole" in tree.groups


def test_experiments_exclude_base_templates(tree):
    assert not any(e.path.name.startswith("_") for e in tree.experiments.values())


# ----------------------------------------------------------- agent rules


def test_online_rl_permits_only_ppo(tree):
    names = {a.name for a in tree.agents_for("online_rl")}
    assert names == {"ppo"}


def test_offline_rl_permits_offline_algorithms(tree):
    names = {a.name for a in tree.agents_for("offline_rl")}
    assert names == {"cql", "iql"}


def test_offline_rl_rejects_ppo(tree):
    assert not tree.paradigms["offline_rl"].permits_agent("ppo")


def test_online_rl_rejects_offline_algorithms(tree):
    online = tree.paradigms["online_rl"]
    for name in ("cql", "iql"):
        assert not online.permits_agent(name)


def test_supervised_has_no_declared_agents(tree):
    """Documents a real gap: supervised declares an empty allow-list."""
    assert tree.agents_for("supervised") == []


def test_permits_agent_accepts_agent_object(tree):
    assert tree.paradigms["online_rl"].permits_agent(tree.agents["ppo"])


def test_agent_allow_and_forbid_lists_are_disjoint(tree):
    for paradigm in tree.paradigms.values():
        assert not set(paradigm.allowed_agents) & set(paradigm.forbidden_agents)


# ------------------------------------------------------- environment rules


def test_online_rl_offers_only_simulator_environments(tree):
    assert all(not e.offline_only for e in tree.environments_for("online_rl"))


def test_offline_rl_offers_only_static_dataset_environments(tree):
    assert all(e.offline_only for e in tree.environments_for("offline_rl"))


def test_online_and_offline_environment_sets_are_disjoint(tree):
    online = {e.name for e in tree.environments_for("online_rl")}
    offline = {e.name for e in tree.environments_for("offline_rl")}
    assert not online & offline


def test_every_environment_is_offered_by_some_paradigm(tree):
    covered = set()
    for name in tree.paradigms:
        covered |= {e.name for e in tree.environments_for(name)}
    # Nested configs are also registered under their file stem (atari/pong as "pong"), so compare
    # the distinct environments, not the dictionary keys.
    assert covered == {e.name for e in tree.environments.values()}


def test_cartpole_is_available_for_online_rl(tree):
    assert "cartpole" in {e.name for e in tree.environments_for("online_rl")}


def test_mimic_is_available_for_offline_rl(tree):
    assert "mimic" in {e.name for e in tree.environments_for("offline_rl")}


def test_paradigm_without_env_constraint_permits_everything():
    paradigm = Paradigm(name="anything", description="", allowed_agents=(), forbidden_agents=())
    assert paradigm.permits_environment(Environment(name="x", offline_only=True))
    assert paradigm.permits_environment(Environment(name="y", offline_only=False))


# ---------------------------------------------------------- field enabling


def test_eval_episodes_enabled_for_online_rl(tree):
    assert tree.field_enabled("online_rl", "eval_episodes")


@pytest.mark.parametrize("paradigm", ["offline_rl", "supervised"])
def test_eval_episodes_disabled_without_rollouts(tree, paradigm):
    assert not tree.field_enabled(paradigm, "eval_episodes")


def test_intervals_enabled_for_online_rl(tree):
    assert tree.field_enabled("online_rl", "intervals_count")


@pytest.mark.parametrize("paradigm", ["offline_rl", "supervised"])
def test_intervals_disabled_without_progressive_slicing(tree, paradigm):
    """validation.py reads allows_intervals with a default of False, so a
    paradigm that stays silent forbids intervals_count > 1."""
    assert not tree.field_enabled(paradigm, "intervals_count")


def test_disabled_field_explains_itself(tree):
    reason = tree.paradigms["offline_rl"].disabled_reason("eval_episodes")
    assert reason and "eval_episodes" in reason


def test_enabled_field_has_no_reason(tree):
    assert tree.paradigms["online_rl"].disabled_reason("eval_episodes") is None


def test_unknown_field_defaults_to_enabled(tree):
    assert tree.field_enabled("online_rl", "some_future_field")


# --------------------------------------------------------------- overrides


def test_overrides_use_hydra_group_names():
    selection = Selection(
        experiment="tests/cartpole", paradigm="online_rl", environment="cartpole", agent="ppo", model="dnn"
    )
    overrides = selection.to_overrides()
    assert "env=cartpole" in overrides
    assert "agent=ppo" in overrides
    assert "model=dnn" in overrides
    assert "paradigm=online_rl" in overrides


def test_site_defaults_to_local():
    assert "site=local" in Selection(experiment="tests/cartpole").to_overrides()


def test_unset_scalars_are_omitted():
    overrides = Selection(experiment="tests/cartpole").to_overrides()
    assert not any(o.startswith("seed=") for o in overrides)


def test_scalars_render_when_set():
    selection = Selection(experiment="tests/cartpole", seed=7, total_timesteps=640)
    overrides = selection.to_overrides()
    assert "seed=7" in overrides
    assert "total_timesteps=640" in overrides


def test_forbidden_scalar_dropped_when_tree_supplied(tree):
    selection = Selection(experiment="tests/mimic", paradigm="offline_rl", eval_episodes=10)
    assert not any(o.startswith("eval_episodes=") for o in selection.to_overrides(tree))


def test_forbidden_scalar_kept_without_tree():
    selection = Selection(experiment="tests/mimic", paradigm="offline_rl", eval_episodes=10)
    assert "eval_episodes=10" in selection.to_overrides()


def test_permitted_scalar_survives_filtering(tree):
    selection = Selection(experiment="tests/cartpole", paradigm="online_rl", eval_episodes=2)
    assert "eval_episodes=2" in selection.to_overrides(tree)


def test_extra_overrides_pass_through_verbatim():
    selection = Selection(experiment="tests/cartpole", extra=["agent.lr=0.007", "no_plot=true"])
    overrides = selection.to_overrides()
    assert "agent.lr=0.007" in overrides
    assert "no_plot=true" in overrides


def test_command_puts_experiment_first():
    argv = Selection(experiment="tests/cartpole").command(script="run_pipeline.py")
    assert argv[1] == "run_pipeline.py"
    assert argv[2] == "tests/cartpole"


def test_command_appends_dry_run_flag():
    argv = Selection(experiment="tests/cartpole").command(dry_run=True)
    assert argv[-1] == "dry_run=true"


def test_command_omits_dry_run_by_default():
    assert "dry_run=true" not in Selection(experiment="tests/cartpole").command()


def test_command_line_is_a_string():
    line = Selection(experiment="tests/cartpole").command_line()
    assert isinstance(line, str)
    assert "run_pipeline.py tests/cartpole" in line
