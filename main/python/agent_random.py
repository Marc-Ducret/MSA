from minecraft_environment import *

env = MinecraftEnv()

while True:
    env.step(env.action_space.sample())
