"""The SB3 runner logs the mean return of every episode that finished between two logs."""
from src.usr.algorithms.sb3_runner import EpisodeRewards


def episode(r):
    return {"episode": {"r": r, "l": 20}}


def test_collects_episodes_from_every_step_not_just_the_logging_step():
    rewards = EpisodeRewards()
    # Episodes end on steps that are not logging steps; the logging step itself finishes none
    for infos in ([episode(10.0)], [{}], [episode(30.0)], [{}]):
        rewards.add(infos)
    assert rewards.mean_and_reset() == 20.0


def test_reset_after_each_log():
    rewards = EpisodeRewards()
    rewards.add([episode(10.0)])
    assert rewards.mean_and_reset() == 10.0
    assert rewards.mean_and_reset() is None  # nothing finished since the last log
    rewards.add([episode(50.0), episode(70.0)])  # vectorised envs can finish several at once
    assert rewards.mean_and_reset() == 60.0


def test_tolerates_missing_infos():
    rewards = EpisodeRewards()
    rewards.add(None)
    assert rewards.mean_and_reset() is None
