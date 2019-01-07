from .sonic_util import AllowBacktracking, make_env

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

env = AllowBacktracking(make_env(stack=False, scale_rew=False))
env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

while not done:
    obs, reward, done, info = env.step(env.action_space.sample)
    print(info['x'])

