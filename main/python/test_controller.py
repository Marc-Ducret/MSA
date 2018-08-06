from minecraft.environment import *
import random
import h5py
import numpy as np

agents_started = False
step_loaded = -1
def step(done):
    global step_loaded
    step_loaded = random.randint(0, 10000)
    return (MinecraftController.LOAD, step_loaded) if agents_started else (MinecraftController.WAIT, None)

N = 4

c = MinecraftController('Pattern', step, 'part1')
c.start()

with h5py.File('tmp/imitation/part1.h5', 'r') as f:
    obs = np.array(f['obsVar'])

envs = [MinecraftEnv(c.env_type, c.env_id, actor_id=actor) for actor in range(N)]
for e in envs:
    e.init_spaces()

agents_started = True
while True:
    ep_done = False
    for e in envs:
        o = e.reset()
        if not np.array_equal(o, obs[step_loaded, e.actor_id]):
            print('obs missmatch')
            dist = np.sum(np.abs(o - obs[step_loaded, e.actor_id]).reshape(24 * 12, 7), axis=0)
            print(dist)
            dist = np.sum(dist)
            for a in range(N):
                d = np.sum(np.abs(o - obs[step_loaded, a]))
                if d < dist:
                    print(a, 'is closer:', d)
    while not ep_done:
        for e in envs:
            e.step_act(e.action_space.sample() * 0)
        for e in envs:
            _, _, done, _ = e.step_result()
            ep_done = done or ep_done
