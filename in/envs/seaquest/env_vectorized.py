from typing import Sequence
from src.app.core.env_vectorized import VectorizedNudgeBaseEnv
from src.app.core.atari_wrappers import make_env
import torch as th
from ocatari.ram.seaquest import MAX_NB_OBJECTS
import gymnasium as gym
from hackatari.core import HackAtari
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

class VectorizedNudgeEnv(VectorizedNudgeBaseEnv):
    name = "seaquest"
    pred2action = {
        "noop": 0,
        "fire": 1,
        "up": 2,
        "right": 3,
        "left": 4,
        "down": 5,
    }

    def __init__(
        self,
        mode: str,
        n_envs: int,
        render_mode="rgb_array",
        render_oc_overlay=False,
        seed=None,
        clip_rewards=False,
    ):
        super().__init__(mode)
        self.n_envs = n_envs
        self.seed = seed
        
        def make_hackatari_env(rank):
            def _thunk():
                env = HackAtari(
                    env_name="ALE/Seaquest-v5",
                    mode="ram",
                    obs_mode="ori",
                    rewardfunc_path="in/envs/seaquest/blenderl_reward.py",
                    render_mode=render_mode,
                    render_oc_overlay=render_oc_overlay,
                )
                env = make_env(env, clip_rewards=clip_rewards)
                # Ensure the top-level observation space matches the wrapped internal space
                # env.observation_space = env._env.observation_space # make_env already handles this
                if seed is not None:
                    env.action_space.seed(seed + rank)
                return env
            return _thunk

        # Use SyncVectorEnv for small number of envs (like evaluation) to save resources
        if n_envs <= 10:
            self.venv = SyncVectorEnv([make_hackatari_env(i) for i in range(n_envs)])
        else:
            import multiprocessing
            import sys
            ctx = None
            if sys.platform == "darwin":
                # Fork is more reliable for inheriting redirected stdout on macOS
                ctx = "fork"
            self.venv = AsyncVectorEnv([make_hackatari_env(i) for i in range(n_envs)], context=ctx)
        
        self.n_actions = 6
        self.n_raw_actions = 18
        self.n_objects = 43
        self.n_features = 4
        
        self.obj_offsets = {}
        offset = 0
        for obj, max_count in MAX_NB_OBJECTS.items():
            self.obj_offsets[obj] = offset
            offset += max_count
        self.relevant_objects = set(MAX_NB_OBJECTS.keys())

    def reset(self):
        obs, infos = self.venv.reset(seed=self.seed)
        # obs is (N, 4, 84, 84)
        neural_states = th.tensor(obs).float()
        
        # Get objects from all envs
        all_objects = self.venv.get_attr("objects")
        
        logic_states = []
        for objects in all_objects:
            logic_states.append(self.extract_logic_state(objects))
        
        return th.stack(logic_states), neural_states

    def step(self, actions, is_mapped: bool = False):
        obs, rewards, terminations, truncations, infos = self.venv.step(actions)
        
        neural_states = th.tensor(obs).float()
        all_objects = self.venv.get_attr("objects")
        
        logic_states = []
        for objects in all_objects:
            logic_states.append(self.extract_logic_state(objects))
            
        return (
            (th.stack(logic_states), neural_states),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def extract_logic_state(self, input_state):
        state = th.zeros((self.n_objects, self.n_features), dtype=th.int32)
        obj_count = {k: 0 for k in MAX_NB_OBJECTS.keys()}
        for obj in input_state:
            if obj.category not in self.relevant_objects:
                continue
            idx = self.obj_offsets[obj.category] + obj_count[obj.category]
            if obj.category == "OxygenBar":
                state[idx] = th.Tensor([1, obj.value, 0, 0])
            else:
                orientation = (
                    obj.orientation.value if obj.orientation is not None else 0
                )
                state[idx] = th.tensor([1, *obj.center, orientation])
            obj_count[obj.category] += 1
        return state

    def extract_neural_state(self, raw_input_state):
        return raw_input_state

    def close(self):
        self.venv.close()
