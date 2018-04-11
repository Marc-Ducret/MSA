import single_env_agent

def run(args, env):
    while True:
        env.reset()
        while True:
            _, _, done, _ = env.step(env.action_space.sample())
            if done:
                break

if __name__ == '__main__':
    single_env_agent.run_agent(run)
