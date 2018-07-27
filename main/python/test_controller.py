from minecraft.environment import *
import random

agents_started = False
def step(done):
    return (MinecraftController.LOAD, random.randint(0, 10000)) if agents_started else (MinecraftController.WAIT, None)

N = 4

c = MinecraftController("Pattern", step, "part1")
c.start()

envs = [MinecraftEnv(c.env_type, c.env_id) for _ in range(N)]
for e in envs:
    e.init_spaces()

agents_started = True
while True:
    ep_done = False
    for e in envs:
        e.reset()
    while not ep_done:
        for e in envs:
            e.step_act(e.action_space.sample())
        for e in envs:
            _, _, done, _ = e.step_result()
            ep_done = done or ep_done
