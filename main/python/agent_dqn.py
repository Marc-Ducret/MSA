import single_env_agent
import numpy as np
import tensorflow as tf
from collections import deque

def run(args, env):
    model = Model(env)
    actions = []
    for i in range(env.action_dim):
        for v in [-1, +1]:
            actions.append(np.array([v if i == j else 0 for j in range(env.action_dim)]))

    with tf.Session() as sess:
        gamma = .99
        batch_size = 64
        num_batches = 32
        learning_rate = 1e-5

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tf_value_batch = tf.placeholder(tf.float32, (batch_size,))
        tf_loss = tf.reduce_mean(tf.squared_difference(tf_value_batch, model.output))
        tf_optimize = optimizer.apply_gradients(optimizer.compute_gradients(tf_loss))

        sess.run(tf.global_variables_initializer())
        episode_reward_buffer = deque(maxlen=100)

        def add_future_rewards(episode):
            for i in reversed(range(len(episode)-1)):
                observation, action, reward, _  = episode[i]
                _, _, reward_next, _ = episode[i+1]
                episode[i] = observation, action, reward + gamma * reward_next, False

        def gather_episode():
            episode = []
            observation = env.reset()
            episode_reward = 0
            while True:
                action = model.act(actions, observation, sess)
                new_observation, reward, done, _ = env.step(action)
                episode.append((observation, action, reward, done))
                episode_reward += reward

                if done:
                    episode_reward_buffer.append(episode_reward)
                    add_future_rewards(episode)
                    return episode

        while True:
            data = []
            while len(data) < num_batches * batch_size:
                data += gather_episode()
                model.epsilon = max(.01, model.epsilon * .999)
            data = np.array(data)
            np.random.shuffle(data)
            for i in range(num_batches):
                observation_batch = np.concatenate([data[j][0].reshape((1, env.observation_dim))
                    for j in range(i*batch_size, (i+1)*batch_size)])
                action_batch = np.concatenate([data[j][1].reshape((1, env.action_dim))
                    for j in range(i*batch_size, (i+1)*batch_size)])
                value_batch = np.concatenate([np.array([data[j][2]])
                    for j in range(i*batch_size, (i+1)*batch_size)])
                loss = sess.run(tf_loss, { model.observation: observation_batch,
                                        model.action: action_batch,
                                        tf_value_batch: value_batch})
                sess.run(tf_optimize, { model.observation: observation_batch,
                                        model.action: action_batch,
                                        tf_value_batch: value_batch})
            print('reward: %f, epsilon: %f, loss: %f' % (np.mean(episode_reward_buffer), model.epsilon, loss))


class Model:
    def __init__(self, env):
        self.hidden_size = 64
        self.hidden_layers = 2
        self.observation = tf.placeholder(tf.float32, (None, env.observation_dim))
        self.action = tf.placeholder(tf.float32, (None, env.action_dim))
        self.input = tf.concat((self.observation, self.action), 1)
        hidden = self.input
        for _ in range(self.hidden_layers):
            hidden = tf.layers.dense(hidden, self.hidden_size, activation=tf.tanh)
        self.output = tf.layers.dense(hidden, 1)
        self.epsilon = 1

    def act(self, actions, observation, sess):
        if np.random.random() < self.epsilon:
            return actions[np.random.randint(len(actions))]
        values = np.array([
            sess.run(self.output,
                {self.observation: observation.reshape((1,) + observation.shape),
                 self.action: action.reshape((1,) + action.shape)})
            for action in actions])

        return actions[np.argmax(values)]

if __name__ == '__main__':
    single_env_agent.run_agent(run)
