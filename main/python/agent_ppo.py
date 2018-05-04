import single_env_agent
from baselines.common import tf_util as U, distributions
from baselines.ppo1 import pposgd_simple
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
import numpy as np

class Policy:

    def __init__(self, name, env, args):
        with tf.variable_scope(name):
            self.env = env
            self._init(env, args)
            self.scope = tf.get_variable_scope().name

    def _init(self, env, args):
        self.recurrent = False
        self.pdtype = distributions.DiagGaussianPdType(env.action_dim)

        ob = U.get_placeholder(name='ob', dtype=tf.float32, shape=(None, env.observation_dim))
        ob_entities = U.get_placeholder(name='ob_entities', dtype=tf.float32,
            shape=(None, env.entity_max, env.entity_dim))
        entity_mask = U.get_placeholder(name='entity_mask', dtype=tf.bool,
            shape=(None, env.entity_max))

        #TODO if env.entity_max == 0
        with tf.variable_scope('attention'):
            ob_entities_masked = tf.boolean_mask(ob_entities, tf.reduce_any(entity_mask, 0), axis=1)
            layer = ob_entities_masked
            ob_ent_shape = tf.shape(layer)
            for i in range(args.hid_layers):
                layer = tf.layers.dense(layer, args.hid_size, name='dense_%i' % i, activation=tf.nn.tanh,
                                kernel_initializer=U.normc_initializer(1))
            attention = tf.layers.dense(layer, 1, name='final',
                            kernel_initializer=U.normc_initializer(1))
            attention = tf.multiply(
                tf.reshape(tf.cast(tf.boolean_mask(entity_mask, tf.reduce_any(entity_mask, 0), axis=1), tf.float32), (ob_ent_shape[0], ob_ent_shape[1], 1)),
                attention)
            attention = tf.nn.softmax(tf.tanh(attention) * 6, axis=1)
            self.attention_entropy = tf.reduce_mean(tf.reduce_sum(- tf.log(attention) * attention, 1))
            self._attention = U.function([ob_entities, entity_mask], [attention, self.attention_entropy])
            ob_attention = attention * ob_entities_masked
            ob_attention = tf.reduce_sum(ob_attention, axis=1)

        #with tf.variable_scope('obfilter'):
        #    self.ob_rms = RunningMeanStd(shape=(env.observation_dim,))

        with tf.variable_scope('vf'):
            #obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5, 5)
            obz = tf.concat([ob, ob_attention], 1)
            layer = obz
            for i in range(args.hid_layers):
                layer = tf.layers.dense(layer, args.hid_size, name='dense_%i' % i, activation=tf.nn.tanh,
                                kernel_initializer=U.normc_initializer(1))
            self.vpred = tf.layers.dense(layer, 1, name='final', kernel_initializer=U.normc_initializer(1))

        with tf.variable_scope('pol'):
            layer = obz
            for i in range(args.hid_layers):
                layer = tf.layers.dense(layer, args.hid_size, name='dense_%i' % i, activation=tf.nn.tanh,
                                kernel_initializer=U.normc_initializer(1))
            pdparam = tf.layers.dense(layer, self.pdtype.param_shape()[0], name='final',
                                kernel_initializer=U.normc_initializer(1))

        self.pd = self.pdtype.pdfromflat(pdparam)

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, ob_entities, entity_mask], [ac, self.vpred])

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('action_dim: %i, observation_dim: %i, entity_dim: %i' %
            (env.action_dim, env.observation_dim, env.entity_dim))
        print('name: ', tf.get_variable_scope().name, 'params:', total_parameters)

    def act(self, stochastic, ob):
        (ob_const, ob_entities, entity_mask) = ob
        ob_entities = ob_entities.reshape((ob_entities.shape[0] // self.env.entity_dim,
            self.env.entity_dim))
        #print(self._attention(ob_entities[None]))
        ac1, vpred1 =  self._act(stochastic, ob_const[None], ob_entities[None], entity_mask[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

def train(args, env):
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return Policy(name, env, args)

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
