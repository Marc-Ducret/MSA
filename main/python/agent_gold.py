import single_env_agent
import numpy as np

def run(args, env):
    def act(obs):
        w = 10
        h = 5
        obs = obs.reshape((h, w))
        fwd = 0
        stf = 0
        yaw = 0
        pch = 0

        m = np.amax(obs)
        if m < .5 and np.mean(obs) < -.5:
            pch = .1
            fwd = -.1
        elif m < .5:
            yaw = .5
        else:
            i = np.argmax(obs)
            x = i % w
            y = i // w
            yaw = .5 * (x - (w-1)/2) / ((w-1)/2)
            pch = .5 * (y - (h-1)/2) / ((h-1)/2)
            fwd = .1
            if np.abs(yaw) + np.abs(pch) < .1:
                fwd = .5

        return np.array([fwd, stf, yaw, pch])
    while True:
        obs, _, _ = env.reset()
        while True:
            (obs, _, _), _, done, _ = env.step(act(obs))
            if done:
                break

if __name__ == '__main__':
    single_env_agent.run_agent(run)
