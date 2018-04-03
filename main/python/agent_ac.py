import utils
import numpy as np
from ac_agent import *

from minecraft_environment import *


def main():
    sess = utils.use_gpu(False)
    env = MinecraftEnv()
    agent = ActorCritic(env.observation_dim, env.action_dim, sess)

    obs = env.reset().reshape((1, env.observation_dim))
    eprint(obs)
    for t in range(10 ** 10):
        action = agent.act(obs).reshape((1, env.action_dim))
        next_obs, reward, done, _ = env.step(action[0])
        next_obs = next_obs.reshape((1, env.observation_dim))
        agent.remember(obs, action, reward, next_obs, done)
        if t % 6 == 0:
            agent.train()
        if t % 60 == 0:
            agent.update_target()
        obs = env.reset().reshape((1, env.observation_dim)) if done else next_obs


if __name__ == '__main__':
    main()
