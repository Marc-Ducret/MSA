from minecraft_environment import *

def train(num_timesteps, seed):
    env = MinecraftEnv()
    params = {'filename': 'model'}
    params.update(env.params())

    from baselines.common import tf_util as U
    from baselines.ppo1 import mlp_policy, pposgd_simple
    import tensorflow as tf

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, './tmp/models/'+params['filename'])

    while True:
        obs = env.reset()
        done = False
        while not done:
            obs, _, done, _ = env.step(pi.act(True, obs)[0])

    env.close()

def main():
    train(num_timesteps=10 ** 5, seed=42)

if __name__ == '__main__':
    main()
