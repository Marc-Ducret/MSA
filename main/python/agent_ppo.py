from minecraft_environment import *

def train():
    env = MinecraftEnv()

    params = {
        'hid_size': 64,
        'hid_layers': 2,
        'total_timesteps': 10 ** 6,
        'actorbatch_timesteps': 2000,
        'clip': 0.2,
        'entropy': 0.0,
        'epochs': 10,
        'learning_rate': 3e-4,
        'batchsize': 64,
        'gamma': .99,
        'lam': .95,
        'schedule': 'linear',
        'filename': 'model',
        }
    params.update(env.params())

    from baselines.common import tf_util as U
    from baselines.ppo1 import mlp_policy, pposgd_simple
    import tensorflow as tf

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=params['hid_size'], num_hid_layers=params['hid_layers'])

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=params['total_timesteps'],
            timesteps_per_actorbatch=params['actorbatch_timesteps'],
            clip_param=params['clip'], entcoeff=params['entropy'],
            optim_epochs=params['epochs'], optim_stepsize=params['learning_rate'], optim_batchsize=params['batchsize'],
            gamma=params['gamma'], lam=params['lam'], schedule=params['schedule'],
        )
    tf.train.Saver().save(sess, './tmp/'+params['filename'])
    env.close()

def main():
    train()

if __name__ == '__main__':
    main()
