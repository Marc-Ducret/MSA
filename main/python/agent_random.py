from minecraft_environment import *

env = MinecraftEnv()
while True:
    env.reset()
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            break
