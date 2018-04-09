from minecraft_environment import *

env = MinecraftEnv()
env.params()

import numpy as np

while True:
    env.reset()
    while True:
        _, _, done, _ = env.step(np.zeros(env.action_space.shape))
        if done:
            break
