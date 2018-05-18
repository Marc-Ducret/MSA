import single_env_agent
import numpy as np

def run(args, env):
    def act(obs):
        w = 24
        h = 12
        obs = obs.reshape((h, w))
        fwd = 0
        stf = 0
        yaw = 0
        pch = 0
        hit = 0

        size = np.mean(np.maximum(obs, 0))

        m = np.amax(obs)
        if m < .5 and np.mean(obs) < -.5:
            pch = .1
            fwd = -.1
        elif m < .5 and np.amin(obs) > -.5:
            pch = -.1
        elif m < .5:
            yaw = .5
        else:
            y, x = np.mean(np.argwhere(obs == m), axis=0)
            yaw = (x - (w-1)/2) / ((w-1)/2)
            pch = (y - (h-1)/2) / ((h-1)/2)
            if size < .1:
                fwd = 1 - min((yaw ** 2 + pch ** 2) * 3, 1)
                fwd *= 1
            elif (yaw ** 2 + pch ** 2) < .03:
                hit = 1
            yaw *= .3
            pch *= .15
        return np.array([fwd, stf, yaw, pch, hit])
    while True:
        obs, _, _ = env.reset()
        while True:
            (obs, _, _), _, done, _ = env.step(act(obs))
            if done:
                break

if __name__ == '__main__':
    single_env_agent.run_agent(run)
