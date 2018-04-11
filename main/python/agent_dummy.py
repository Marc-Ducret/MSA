import single_env_agent
import numpy as np

def run(args, env):
    while True:
        env.reset()
        while True:
            _, _, done, _ = env.step(np.zeros(env.action_space.shape))
            if done:
                break

if __name__ == '__main__':
    single_env_agent.run_agent(run)
