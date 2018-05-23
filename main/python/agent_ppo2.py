from minecraft.environment import *
from baselines.common import set_global_seeds
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import concurrent.futures
import gym
import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

class _ParEnv:

    def __init__(self, env_type, num):
        self.envs = [MinecraftEnv(env_type) for _ in range(num)]
        for e in self.envs:
            e.init_spaces()
        self.num_envs = num
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.buf_obs = np.zeros((self.num_envs,) + tuple(self.observation_space.shape), self.observation_space.dtype)
        self.buf_done = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_reward = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_info = [{} for _ in range(self.num_envs)]

    def step(self, actions):
        for i in range(self.num_envs):
            self.envs[i].step_act(actions[i])

        for i in range(self.num_envs):
            obs, reward, done, info = self.envs[i].step_result()
            if done:
                obs = self.envs[i].reset()
            self.buf_obs[i] = obs
            self.buf_reward[i] = reward
            self.buf_done[i] = done
            self.buf_info[i] = info

        return self.buf_obs, self.buf_reward, self.buf_done, self.buf_info

    def reset(self):
        return [e.reset() for e in self.envs]

    def close(self):
        for e in self.envs:
            e.close()

def train(env_type):
    N = 16
    env = _ParEnv(env_type, N)

    ncpu = 4
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    set_global_seeds(42)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=256, nminibatches=64,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=-0.0005,
        lr=lambda t: 1e-4 * t,
        cliprange=0.2,
        total_timesteps=10 ** 6, vf_coef=.5)

    env.close()

def main():
    train('ParkourRandom')

if __name__ == '__main__':
    main()
