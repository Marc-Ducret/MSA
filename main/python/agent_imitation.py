import tensorflow as tf
import numpy as np
import h5py
from baselines.common import tf_util as U, distributions

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

def train(obs_dataset, act_dataset):
    n = obs_dataset.shape[0] * obs_dataset.shape[1]
    obs_dim = obs_dataset.shape[2]
    act_dim = act_dataset.shape[2]
    obs_dataset = obs_dataset.reshape((n, obs_dim))
    act_dataset = act_dataset.reshape((n, act_dim))
    obs_in, act_in, act_out = policy(obs_dim, act_dim, 64, 2)

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
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(100):
            for batch in epoch(32):
                obs_batch, act_batch = batch
                sess.run(optimize, feed_dict={obs_in: obs_batch, act_in: act_batch})
            losses = []
            for batch in epoch(32):
                obs_batch, act_batch = batch
                losses.append(sess.run(loss, feed_dict={obs_in: obs_batch, act_in: act_batch}))
            print('epoch %i: loss=%f' % (e+1, np.mean(np.array(losses))))


def main():
    f = h5py.File('tmp/imitation/dataset.h5')
    train(np.array(f['obsVar']), np.array(f['actVar']))

if __name__ == '__main__':
    main()
