import gymnasium as gym

from . import agents
from .dynamic_catch_v1 import DynamicCatchEnvV1, DynamicCatchEnvCfgV1
from .dynamic_catch_v2 import DynamicCatchEnvV2, DynamicCatchEnvCfgV2

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Dynamic-Catch-rl_games",
    entry_point="omni.isaac.lab_tasks.direct.catchpolicy.dynamic_catch_v1:DynamicCatchEnvV1",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DynamicCatchEnvCfgV1,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_separate_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Dynamic-Catch-Goal-Estimator",
    entry_point="omni.isaac.lab_tasks.direct.catchpolicy.dynamic_catch_v2:DynamicCatchEnvV2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DynamicCatchEnvCfgV2,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_separate_cfg.yaml"
    }
)