import single_env_agent
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import tensorflow as tf

def train(args, env):
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=args.hid_size, num_hid_layers=args.hid_layers)

    pposgd_simple.learn(env, policy_fn,
            max_timesteps=args.total_timesteps,
            timesteps_per_actorbatch=args.actorbatch_timesteps,
            clip_param=args.clip, entcoeff=args.entropy,
            optim_epochs=args.epochs, optim_stepsize=args.learning_rate, optim_batchsize=args.batchsize,
            gamma=args.gamma, lam=args.lam, schedule=args.schedule,
        )
    tf.train.Saver().save(sess, './tmp/models/'+args.filename)

def main():
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
    single_env_agent.run_agent(train, params)

if __name__ == '__main__':
    main()
