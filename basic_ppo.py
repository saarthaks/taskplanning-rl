import gym
import gym_basic_nav

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('continuous-nav-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=200)

obs = env.reset()
reward = 0
actions = 0*env.action_space.sample()
states = []
for i in range(500):
    action, _states = model.predict(obs)
    actions += action.squeeze()
    obs, rewards, dones, info = env.step(action)
    reward += rewards
    states.append(obs[0,:2])
    env.render()
    # print("Reward: %d" % rewards)
    # print("Total: %d\n" % reward)

    if dones:
        print("COMPLETE\n")
        break

print(states[:10])
print("Total: %d\n" % reward)
