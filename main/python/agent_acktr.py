import tensorflow as tf
from baselines import logger
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
import single_env_agent
from argparse import Namespace

def train(args, env):
    env.spec = Namespace()
    env.spec.timestep_limit = 1000
    logger.configure('./tmp/logs/', ['tensorboard', 'stdout'])

    with tf.Session(config=tf.ConfigProto()):
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)

        learn(env, policy=policy, vf=vf,
            gamma=args.gamma, lam=args.lam, timesteps_per_batch=args.actorbatch_timesteps,
            desired_kl=args.desired_kl,
            num_timesteps=args.total_timesteps, animate=False)

def main():
    params = {
        'total_timesteps': 10 ** 6,
        'actorbatch_timesteps': 2000,
        'gamma': .99,
        'lam': .95,
        'desired_kl': .002
        }
    single_env_agent.run_agent(train, params)

if __name__ == '__main__':
    main()
