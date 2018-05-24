import tensorflow as tf
import numpy as np
import h5py
import plotly.plotly as py
import plotly.graph_objs as go
import single_env_agent
import minecraft.environment
from collections import deque
from time import time
import concurrent.futures
import traceback

def size(dim, ratio):
    w = int(np.sqrt(dim * ratio))
    h = w // ratio
    return w, h

def policy(obs_dim, act_dim, hid_layers, hid_size):
    obs_in = tf.placeholder(tf.float32, shape=(None, obs_dim), name='observation')
    act_in = tf.placeholder(tf.float32, shape=(None, act_dim), name='action')
    with tf.variable_scope('policy'):
        layer = obs_in
        for i in range(hid_layers):
            layer = tf.layers.dense(layer, hid_size, name='dense_%i' % i,
                        activation=tf.nn.tanh)
        act_out = tf.layers.dense(layer, act_dim, name='action')
        return obs_in, act_in, act_out

def policy_cnn(obs_dim_w, obs_dim_h, act_dim):
    obs_in = tf.placeholder(tf.float32, shape=(None, obs_dim_h, obs_dim_w, 1), name='observation')
    act_in = tf.placeholder(tf.float32, shape=(None, act_dim), name='action')

    with tf.variable_scope('policy'):
        layer = tf.nn.depthwise_conv2d(
            obs_in,
            tf.Variable(np.random.normal(size=3*3*3).astype('float32').reshape((3, 3, 1, 3))),
            [1, 1, 1, 1],
            'SAME'
        )
        layer = tf.nn.depthwise_conv2d(
            layer,
            tf.Variable(np.random.normal(size=3*3*3*3).astype('float32').reshape((3, 3, 3, 3))),
            [1, 1, 1, 1],
            'SAME'
        )
        layer = tf.nn.depthwise_conv2d(
            layer,
            tf.Variable(np.random.normal(size=3*3*9).astype('float32').reshape((3, 3, 9, 1))),
            [1, 1, 1, 1],
            'SAME'
        )
        layer = tf.nn.max_pool(
            layer,
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            'SAME'
        )
        layer = tf.nn.depthwise_conv2d(
            layer,
            tf.Variable(np.random.normal(size=3*3*9).astype('float32').reshape((3, 3, 9, 1))),
            [1, 1, 1, 1],
            'SAME'
        )
        layer = tf.nn.max_pool(
            layer,
            [1, 3, 3, 1],
            [1, 3, 3, 1],
            'SAME'
        )
        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 64, activation=tf.nn.tanh)
        layer = tf.layers.dense(layer, 64, activation=tf.nn.tanh)
        act_out = tf.layers.dense(layer, act_dim, name='action')
        return obs_in, act_in, act_out

def train(obs_dataset, act_dataset, policy):
    n = obs_dataset.shape[0] * obs_dataset.shape[1]
    obs_dim = obs_dataset.shape[2]
    act_dim = act_dataset.shape[2]
    w, h = size(obs_dim, 2)
    obs_dataset = obs_dataset.reshape((n, h, w, 1))
    act_dataset = act_dataset.reshape((n, act_dim))
    obs_in, act_in, act_out = policy

    def shuffle(a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def epoch(batch_size):
        shuffle(obs_dataset, act_dataset)
        for i in range(n // batch_size):
            obs_batch = obs_dataset[i * batch_size : (i+1) * batch_size]
            act_batch = act_dataset[i * batch_size : (i+1) * batch_size]
            yield obs_batch, act_batch

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    loss = tf.losses.mean_squared_error(act_in, act_out)
    optimize = optimizer.apply_gradients(optimizer.compute_gradients(loss))
    def compute_loss():
        losses = []
        for batch in epoch(1024):
            obs_batch, act_batch = batch
            losses.append(sess.run(loss, feed_dict={obs_in: obs_batch, act_in: act_batch}))
        return np.mean(np.array(losses))

    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            sess.run(tf.global_variables_initializer())
            losses = [compute_loss()]
            rewards_futures = []
            epochs = 20
            batchsize = 64
            print('initial loss=%f' % losses[-1])
            for e in range(epochs):
                start_time = time()
                pairs = 0
                for batch in epoch(batchsize):
                    obs_batch, act_batch = batch
                    sess.run(optimize, feed_dict={obs_in: obs_batch, act_in: act_batch})
                    pairs += batchsize
                speed = pairs / (time() - start_time)
                losses.append(compute_loss())
                print('epoch %i: \tloss=%.4f \tspeed=%.1f' % (e, losses[-1], speed))
                if e % 1 == 0:
                    print('Model saved at: %s' % saver.save(sess, 'tmp/models/imitation_epoch_%i.ckpt' % e))
                    rewards_futures.append(executor.submit(estimate_reward, e))
            trace_loss = go.Scatter(x=list(range(epochs+1)), y=losses)
            trace_rew  = go.Scatter(x=list(range(epochs)), y=[f.result() for f in rewards_futures])
            py.iplot([trace_rew], filename='reward')
            py.iplot([trace_loss], filename='loss')

def main():
    f = h5py.File('tmp/imitation/dataset.h5')
    obs_dataset = np.array(f['obsVar'])
    act_dataset = np.array(f['actVar'])
    obs_dim = obs_dataset.shape[2]
    act_dim = act_dataset.shape[2]
    w, h = size(obs_dim, 2)
    train(obs_dataset, act_dataset, policy_cnn(w, h, act_dim))

def play(args, env):
    w, h = size(env.observation_dim, 2)
    obs_in, act_in, act_out = policy_cnn(w, h, env.action_dim)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'tmp/models/imitation_epoch_%i.ckpt' % args.epoch)
        def act(obs):
            action = sess.run(act_out, feed_dict={obs_in: obs.reshape((1, h, w, 1))})
            return action
        eps_rew = deque(maxlen=10000)
        while True:
            obs = env.reset()
            ep_rew = 0
            while True:
                obs, rew, done, _ = env.step(act(obs))
                ep_rew += rew
                if done:
                    eps_rew.append(ep_rew)
                    print('rew=%.2f \tmean=%.2f \t(ct=%i)' % (ep_rew, np.mean(eps_rew), len(eps_rew)))
                    break

def estimate_reward(epoch):
    try:
        tf.reset_default_graph()
        print('estimating %i...' % epoch)
        env = minecraft.environment.MinecraftEnv('GoBreakGold')
        env.init_spaces()
        print('env started....')
        n_eps = 1000
        w, h = size(env.observation_dim, 2)
        obs_in, act_in, act_out = policy_cnn(w, h, env.action_dim)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
            saver.restore(sess, 'tmp/models/imitation_epoch_%i.ckpt' % epoch)
            def act(obs):
                action = sess.run(act_out, feed_dict={obs_in: obs.reshape((1, h, w, 1))})
                return action
            mean_rew = 0
            for i in range(n_eps):
                obs = env.reset()
                ep_rew = 0
                while True:
                    obs, rew, done, _ = env.step(act(obs))
                    ep_rew += rew
                    if done:
                        ep_rew = 100 if ep_rew > 0 else 0
                        mean_rew += ep_rew / n_eps
                        break
            print('estimated %.2f for epoch %i' % (mean_rew, epoch))
            return mean_rew
    except:
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        single_env_agent.run_agent(play, {'epoch': 0})
    else:
        main()
