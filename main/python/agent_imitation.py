import tensorflow as tf
import numpy as np
import h5py
from baselines.common import tf_util as U, distributions
import plotly.plotly as py
import plotly.graph_objs as go
import single_env_agent

def policy(obs_dim, act_dim, hid_layers, hid_size):
    obs_in = tf.placeholder(tf.float32, shape=(None, obs_dim), name='observation')
    act_in = tf.placeholder(tf.float32, shape=(None, act_dim), name='action')
    with tf.variable_scope('policy'):
        layer = obs_in
        for i in range(hid_layers):
            layer = tf.layers.dense(layer, hid_size, name='dense_%i' % i,
                        activation=tf.nn.tanh,
                        kernel_initializer=U.normc_initializer(1))
        act_out = tf.layers.dense(layer, act_dim, name='action',
                        kernel_initializer=U.normc_initializer(1))
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
            tf.Variable(np.random.normal(size=3*3*3).astype('float32').reshape((3, 3, 3, 1))),
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
    obs_dataset = obs_dataset.reshape((n, 6, 12, 1))
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

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    loss = tf.losses.mean_squared_error(act_in, act_out)
    optimize = optimizer.apply_gradients(optimizer.compute_gradients(loss))
    def compute_loss():
        losses = []
        for batch in epoch(32):
            obs_batch, act_batch = batch
            losses.append(sess.run(loss, feed_dict={obs_in: obs_batch, act_in: act_batch}))
        return np.mean(np.array(losses))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = [compute_loss()]
        epochs = 40
        print('initial loss=%f' % losses[-1])
        for e in range(epochs):
            for batch in epoch(32):
                obs_batch, act_batch = batch
                sess.run(optimize, feed_dict={obs_in: obs_batch, act_in: act_batch})
            losses.append(compute_loss())
            print('epoch %i: loss=%f' % (e+1, losses[-1]))
            if e % 5 == 0:
                print('Model saved at: %s' % saver.save(sess, 'tmp/models/imitation_epoch_%i.ckpt' % e))

        print('Model saved at: %s' % saver.save(sess, 'tmp/models/imitation_final.ckpt'))
        trace = go.Scatter(x=list(range(epochs+1)), y=losses)
        py.iplot([trace], filename='basic-line')

def main():
    #N = 1000
    #x = np.linspace(0, 3, num=N)
    #y = np.sin(x)
    #train(x.reshape((N, 1, 1)), y.reshape((N, 1, 1)))
    f = h5py.File('tmp/imitation/dataset.h5')
#    def simple(obs):
#        s = np.sqrt(obs.shape[0])
#        i = np.argmax(obs)
#        x, y = (i%s) - (s+1)/2, (i//s) - (s+1)/2
#        return [[y, x]] / ((s-1)/2)

    #dots = np.array([np.dot(simple(f['obsVar'][i][0])[0], f['actVar'][i][0]) for i in range(len(f['obsVar']))])
    #print(np.mean(dots))
    train(np.array(f['obsVar']), np.array(f['actVar']), policy_cnn(12, 6, np.array(f['actVar']).shape[2]))

def play(args, env):
    obs_in, act_in, act_out = policy_cnn(12, 6, env.action_dim)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'tmp/models/imitation_epoch_%i.ckpt' % args.epoch)
        def act(obs):
            action = sess.run(act_out, feed_dict={obs_in: obs.reshape((1, 6, 12, 1))})
            print(action)
            return action
        while True:
            obs, _, _ = env.reset()
            while True:
                (obs, _, _), _, done, _ = env.step(act(obs))
                if done:
                    break


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        single_env_agent.run_agent(play, {'epoch': 0})
    else:
        main()
