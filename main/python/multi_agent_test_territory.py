from minecraft.environment import *
import concurrent.futures
import sys

N = 5
if len(sys.argv) > 1:
    N = int(sys.argv[1])

dummy = MinecraftEnv('Territory')
dummy.init_spaces()
dummy.close()

envs = [MinecraftEnv(dummy.env_type, dummy.env_id) for _ in range(N)]
for env in envs:
    env.init_spaces()

while True:
    for env in envs:
        env.step_act(env.action_space.sample())
    for env in envs:
        ob, reward, done, info = env.step_result()
        if done:
            ob = env.reset()
