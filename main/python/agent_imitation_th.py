import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
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
import copy
from rl.utils import AddBias
from rl.distributions import FixedNormal

def size(dim, ratio):
    w = int(np.sqrt(dim * ratio))
    h = w // ratio
    return w, h

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class DiagGaussian(nn.Module):
    def __init__(self, dim):
        super(DiagGaussian, self).__init__()

        self.logstd = AddBias(torch.zeros(dim))

    def forward(self, x):
        zeros = torch.zeros(x.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(x, action_logstd.exp())

class Policy(nn.Module):
    def __init__(self, obs_dim_w, obs_dim_h, act_dim):
        super(Policy, self).__init__()
        def init_(m):
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight)
            return m

        self.vision = nn.Sequential(
            nn.ReflectionPad2d(1),
            init_(nn.Conv2d(1, 3, 3, groups=1, bias=False)),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            init_(nn.Conv2d(3, 9, 3, groups=3, bias=False)),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            init_(nn.Conv2d(9, 9, 3, groups=9, bias=False)),
            nn.MaxPool2d(2),
            nn.ReflectionPad2d(1),
            init_(nn.Conv2d(9, 9, 3, groups=9, bias=False)),
            nn.MaxPool2d(3),
            Flatten(),

        ).cuda()

        self.feature_dim = 9 * 2 * 4

        self.action = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        ).cuda()

        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).cuda()

        self.dist = DiagGaussian(act_dim).cuda()

        self.train()

    def act(self, inputs, states=None, masks=None):
        features = self.vision(inputs)
        value, actor_features = self.critic(features), self.action(features)
        dist = self.dist(actor_features)

        action = actor_features#dist.rsample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states=None, masks=None):
        return self.critic(self.vision(inputs))

    def evaluate_actions(self, inputs, states, masks, action):
        features = self.vision(inputs)
        value, actor_features = self.critic(features), self.action(features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


def train(obs_dataset, act_dataset, policy):
    optimizer = optim.Adam(policy.parameters(), lr=5e-6)

    def epoch(batch_size, n=None):
        n = n if n else obs_dataset.size(0)
        ids = th.randperm(n).cuda()
        for batch_ids in ids.split(batch_size):
            yield obs_dataset[batch_ids], act_dataset[batch_ids]

    def compute_loss():
        with th.no_grad():
            losses = []
            for obs_batch, act_batch in epoch(1024 * 16):
                _, action, _, _ = policy.act(obs_batch)
                losses.append(F.mse_loss(act_batch, action).cpu().detach().numpy())
            return np.mean(np.array(losses))

    def opt():
        pairs = 0
        for obs_batch, act_batch in epoch(64):
            _, action, _, _ = policy.act(obs_batch)
            loss = F.mse_loss(act_batch, action)
            loss.backward()
            optimizer.step()
            pairs += obs_batch.shape[0]
        return pairs

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        losses = [compute_loss()]
        rewards_futures = []
        epochs = 20
        print('initial loss=%f' % losses[-1])
        for e in range(epochs):
            start_time = time()
            pairs = opt()
            speed = pairs / (time() - start_time)
            losses.append(compute_loss())
            print('epoch %i: \tloss=%.4f \tspeed=%.1f' % (e, losses[-1], speed))
            if e % 1 == 0:
                th.save(policy, 'tmp/models/imitation_th_epoch_%i.pt' % e)
                rewards_futures.append(executor.submit(estimate_reward, e, copy.deepcopy(policy)))
        trace_loss = go.Scatter(x=list(range(epochs+1)), y=losses)
        trace_rew  = go.Scatter(x=list(range(epochs)), y=[f.result() for f in rewards_futures])
        py.iplot([trace_rew], filename='reward')
        py.iplot([trace_loss], filename='loss')

def main():
    f = h5py.File('tmp/imitation/dataset.h5')
    obs_dataset = np.array(f['obsVar'])
    act_dataset = np.array(f['actVar'])
    n = obs_dataset.shape[0] * obs_dataset.shape[1]
    obs_dim = obs_dataset.shape[2]
    act_dim = act_dataset.shape[2]
    w, h = size(obs_dim, 2)
    train(
        th.from_numpy(obs_dataset.reshape((n, h, w, 1)).transpose(0, 3, 1, 2)).cuda(),
        th.from_numpy(act_dataset.reshape((n, act_dim))).cuda(),
        Policy(w, h, act_dim)
    )

"""def play(args, env):
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
            obs, _, _ = env.reset()
            ep_rew = 0
            while True:
                (obs, _, _), rew, done, _ = env.step(act(obs))
                ep_rew += rew
                if done:
                    eps_rew.append(ep_rew)
                    print('rew=%.2f \tmean=%.2f \t(ct=%i)' % (ep_rew, np.mean(eps_rew), len(eps_rew)))
                    break"""

def estimate_reward(epoch, policy):
    try:
        print('estimating %i...' % epoch)
        env = minecraft.environment.MinecraftEnv('GoBreakGold')
        env.init_spaces()
        n_eps = 400
        w, h = size(env.observation_dim, 2)
        def act(obs):
            with th.no_grad():
                _, action, _, _ = policy.act(th.from_numpy(obs.reshape((1, h, w, 1)).transpose(0, 3, 1, 2)).float().cuda())
                action = action.cpu().detach().numpy()
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
        #single_env_agent.run_agent(play, {'epoch': 0})
        pass
    else:
        main()
