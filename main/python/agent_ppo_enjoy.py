import single_env_agent
from baselines.common import tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import tensorflow as tf

def run(args, env):
    params = {'filename': 'model'}

    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, './tmp/models/'+args.filename)

    while True:
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs, rew, done, _ = env.step(pi.act(True, obs)[0])
            total_reward += rew
        print('Episode done with reward:', total_reward)

def main():
    params = {'filename': 'model'}
    single_env_agent.run_agent(run, params)

if __name__ == '__main__':
    main()
