from minecraft_environment import *

def train(num_timesteps, seed):
    env = MinecraftEnv()

    from baselines.common import tf_util as U
    from baselines.ppo1 import mlp_policy, pposgd_simple
    import tensorflow as tf

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2000,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='constant',
        )
    tf.train.Saver().save(sess, './tmp/model')
    env.close()

def main():
    train(num_timesteps=10 ** 5, seed=42)

if __name__ == '__main__':
    main()
