import single_env_agent
import numpy as np
import tensorflow as tf
from collections import deque

def run(args, env):
    model = Model(env)
    target = model
    actions = []
    for i in range(env.action_dim):
        for v in [-1, +1]:
            actions.append(np.array([v if i == j else 0 for j in range(env.action_dim)]))

    with tf.Session() as sess:
        gamma = .99
        batch_size = 64
        num_batches = 32
        learning_rate = 1e-3

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tf_value_batch = tf.placeholder(tf.float32, (batch_size, len(actions)))
        tf_loss = tf.reduce_mean(tf.squared_difference(tf_value_batch, target.output))
        tf_optimize = optimizer.apply_gradients(optimizer.compute_gradients(tf_loss))

        sess.run(tf.global_variables_initializer())
        episode_reward_buffer = deque(maxlen=100)

        def add_future_rewards(data):
            for i in range(len(data)):
                observation, action, reward, done, next_observation = data[i]
                if not done:
                    best_Q = np.max(target.values(actions, next_observation, sess))
                    data[i] = observation, action, reward + gamma * best_Q, False, next_observation

        def gather_episode():
            episode = []
            observation = env.reset()
            episode_reward = 0
            while True:
                action = model.act(actions, observation, sess)
                #print('observation', observation)
                #print('values', model.values(actions, observation, sess))
                #print('action:', action, actions[action])
                new_observation, reward, done, _ = env.step(actions[action])
                episode.append((observation, action, reward, done, new_observation))
                episode_reward += reward
                observation = new_observation

                if done:
                    episode_reward_buffer.append(episode_reward)
                    return episode

        while True:
            data = []
            while len(data) < num_batches * batch_size:
                data += gather_episode()
                model.epsilon = max(.01, model.epsilon * .999)

            data = np.array(data)
            np.random.shuffle(data)
            losses = np.zeros(num_batches)
            for i in range(num_batches):
                batch = data[i*batch_size : (i+1)*batch_size]
                add_future_rewards(batch)
                observation_batch = np.concatenate([batch[j][0].reshape((1, env.observation_dim))
                    for j in range(batch_size)])
                value_batch = np.concatenate([model.values(actions, batch[j][0], sess).reshape((1, len(actions)))
                    for j in range(batch_size)])
                for j in range(batch_size):
                    value_batch[j][batch[j][1]] = batch[j][2]
                losses[i] = sess.run(tf_loss, { model.observation: observation_batch,
                                        tf_value_batch: value_batch})
                sess.run(tf_optimize, { model.observation: observation_batch,
                                        tf_value_batch: value_batch})
            print('reward: %f, epsilon: %f, loss %f' % (np.mean(episode_reward_buffer), model.epsilon, np.mean(losses)))
            obs = np.random.random(env.observation_dim) - .5
            print('observation', obs)
            print('values', model.values(actions, obs, sess))


class Model:
    def __init__(self, env):
        self.hidden_size = 24
        self.hidden_layers = 1
        self.observation = tf.placeholder(tf.float32, (None, env.observation_dim))
        self.input = self.observation
        hidden = self.input
        for _ in range(self.hidden_layers):
            hidden = tf.layers.dense(hidden, self.hidden_size, activation=tf.nn.relu)
        self.output = tf.layers.dense(hidden, 2 * env.action_dim)
        self.epsilon = 1

    def values(self, actions, observation, sess):
        return sess.run(self.output,
                {self.observation: observation.reshape((1,) + observation.shape)}).reshape((len(actions),))

    def act(self, actions, observation, sess):
        if np.random.random() < self.epsilon:
            if abs(observation[0]) > abs(observation[1]):
                return 2 if observation[0] < 0 else 3
            else:
                return 0 if observation[1] < 0 else 1
            return np.random.randint(len(actions))
        values = self.values(actions, observation, sess)
        return np.argmax(values)

if __name__ == '__main__':
    single_env_agent.run_agent(run)
