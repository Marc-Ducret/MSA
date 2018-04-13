from minecraft_environment import *
import concurrent.futures

N = 40

dummy = MinecraftEnv('Territory')
dummy.init_spaces()
dummy.close()

envs = [MinecraftEnv(dummy.env_type, dummy.env_id) for _ in range(N)]
for env in envs:
    env.init_spaces()

while True:
    def step_env(env, action, i):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info, i

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for obs, reward, done, info, i in executor.map(
                                lambda i: step_env(envs[i], envs[i].action_space.sample(), i),
                                range(N)):
            pass
