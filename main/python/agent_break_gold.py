import single_env_agent
import numpy as np

def run(args, env):
    def act(obs):
        w = 12
        h = 6
        obs = obs.reshape((h, w))
        hit = 0
        yaw = 0
        pch = 0

        m = np.amax(obs)
        if m < .5 and np.mean(obs) < -.5:
            pch = .1
        elif m < .5:
            yaw = .5
        else:
            i = np.argmax(obs)
            x = i % w
            y = i // w
            yaw = (x - (w-1)/2) / ((w-1)/2)
            pch = (y - (h-1)/2) / ((h-1)/2)
            yaw *= .5
            pch *= .2
            if obs[h//2][w//2] > .5:
                hit = 1
        return np.array([hit, yaw, pch])
    while True:
        obs, _, _ = env.reset()
        while True:
            (obs, _, _), _, done, _ = env.step(act(obs))
            if done:
                break

if __name__ == '__main__':
    single_env_agent.run_agent(run)
