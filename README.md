# Humanoid_Gym
A humanoid learns to find its goal using reinforcement learning in a Unity3D simulation

The following code can be used to test the trained Humanoid Agent
=======================================
```python
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
channel = EngineConfigurationChannel()
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import time,os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy

env_name = "./Humanoid.exe"

env = UnityEnvironment(env_name,seed=1, side_channels=[channel])
channel.set_configuration_parameters(time_scale = 0.4)
# env= UnityToGymWrapper(env, uint8_visual=True)
env= UnityToGymWrapper(env, uint8_visual=False)

env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


model = PPO.load("PPO_unity_humanoid120")
# evaluate_policy()

# mean_reward, std_reward = evaluate_policy(model, model.get_env(),n_eval_episodes=10)

obs= env.reset()

for i in range(1000):
    action, states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

```
