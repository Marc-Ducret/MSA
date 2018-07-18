import single_env_agent
import numpy as np
from collections import deque

def run(args, env):
    def act(obs):
        w = 24
        h = 12
        c = 7
        obs = obs.reshape((h, w, c))
        fwd = 0
        stf = 0
        yaw = 0
        pch = 0
        hit = 0
        place = 0

        o_sky   = obs[:,:,1]
        o_tree  = obs[:,:,2]

        sky_amount = np.mean(o_sky)
        has_tree = np.amax(o_tree) > 0
        if sky_amount > .5:
            pch = 1
        elif sky_amount < .1:
            pch = -1
            fwd = -.05
        elif not has_tree:
            yaw = 1
        else:
            y, x = np.mean(np.argwhere(o_tree == 1), axis=0)
            yaw = (x-w/2) / (w/2)
            if abs(yaw) < .2:
                fwd = 1

        hit = 1

        """size = np.mean(np.maximum(obs, 0))

        m = np.amax(obs)
        if m < .5 and np.mean(obs) < -.5:
            pch = .1
            fwd = -.1
        elif m < .5 and np.amin(obs) > -.5:
            pch = -.1
        elif m < .5:
            yaw = .5
        else:
            y, x, z = np.mean(np.argwhere(obs == m), axis=0)
            yaw = (x - (w-1)/2) / ((w-1)/2)
            pch = (y - (h-1)/2) / ((h-1)/2)
            if size < .1:
                fwd = 1 - min((yaw ** 2 + pch ** 2) * 3, 1)
                fwd *= 1
            elif (yaw ** 2 + pch ** 2) < .03:
                hit = 1
            yaw *= .3
            pch *= .15"""
        return np.array([fwd, stf, yaw, pch, hit, place], 'float64')
    sucesses = deque(maxlen=10000)
    while True:
        obs = env.reset()
        ep_rew = 0
        noise = .1
        while True:
            action = act(obs)
            action += noise * np.random.normal(size=6)
            obs, rew, done, _ = env.step(action)
            ep_rew += rew
            if done:
                success = 100 if ep_rew > 0 else 0
                sucesses.append(success)
                print('rew=%.2f \tmean=%.2f \t(ct=%i)' % (ep_rew, np.mean(sucesses), len(sucesses)))
                break

if __name__ == '__main__':
    single_env_agent.run_agent(run)
