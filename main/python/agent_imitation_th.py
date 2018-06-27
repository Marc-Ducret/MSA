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
import visdom
import argparse

def size(dim, ratio, channels):
    w = int(np.sqrt(dim * ratio / channels))
    h = w // ratio
    return w, h, channels

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

        action_logstd = self.logstd(zeros) - 3
        return FixedNormal(x, action_logstd.exp())

class Policy(nn.Module):
    def __init__(self, obs_dim_w, obs_dim_h, obs_dim_c, act_dim):
        super(Policy, self).__init__()
        def init_(m):
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight)
            return m
        C = 8
        self.vision = nn.Sequential(
            nn.ReflectionPad2d(3),
            init_(nn.Conv2d(obs_dim_c, C, 7, groups=1, bias=False)),
            nn.Tanh(),
            nn.ReflectionPad2d(2),
            init_(nn.Conv2d(C, C, 5, groups=C, bias=False)),
            nn.Tanh(),
            nn.ReflectionPad2d(1),
            init_(nn.Conv2d(C, C, 3, groups=C, bias=False)),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.ReflectionPad2d(1),
            init_(nn.Conv2d(C, C, 3, groups=C, bias=False)),
            nn.Tanh(),
            nn.MaxPool2d(3),
            Flatten(),

        ).cuda()

        self.feature_dim = C * 2 * 4

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

        self.state_size = 1

    def _adapt_inputs(inputs):
        if len(inputs.shape) < 4:
            inputs = inputs.view(-1, 12, 24, 1).permute(0, 3, 1, 2)
        return inputs

    def act(self, inputs, states=None, masks=None, deterministic=False):
        inputs = Policy._adapt_inputs(inputs)
        features = self.vision(inputs)
        value, actor_features = self.critic(features), self.action(features)
        dist = self.dist(actor_features)

        action = actor_features if deterministic else dist.rsample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, states

    def get_value(self, inputs, states=None, masks=None):
        inputs = Policy._adapt_inputs(inputs)
        return self.critic(self.vision(inputs))

    def evaluate_actions(self, inputs, states, masks, action):
        inputs = Policy._adapt_inputs(inputs)
        features = self.vision(inputs)
        value, actor_features = self.critic(features), self.action(features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


def train(obs_dataset, act_dataset, policy, args):
    #PARAMS
    lr = args.lr
    batch_size = args.batch
    loss_function = F.mse_loss
    decay = args.decay

    vis = visdom.Visdom()

    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=decay)
    n = obs_dataset.size(0)

    def epoch(batch_size, size=None, offset=None):
        size = size if size else n
        ids = th.randperm(size).cuda()
        if offset:
            ids += offset
        for batch_ids in ids.split(batch_size):
            yield obs_dataset[batch_ids], act_dataset[batch_ids]

    def compute_loss():
        with th.no_grad():
            losses = []
            for obs_batch, act_batch in epoch(1024 * 16, n, 0):
                _, action, _, _ = policy.act(obs_batch, deterministic=True)
                losses.append(loss_function(act_batch, action).cpu().detach().numpy())
            return np.mean(np.array(losses))

    def opt():
        pairs = 0
        for obs_batch, act_batch in epoch(batch_size, n):
            _, action, _, _ = policy.act(obs_batch, deterministic=True)
            loss = loss_function(act_batch, action)
            loss.backward()
            optimizer.step()
            pairs += obs_batch.shape[0]
        return pairs

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        initial_loss = compute_loss()
        opts = dict(
            title='data=%s lr=%.1e bs=%i dc=%.1e' % (args.dataset, lr, batch_size, decay),
            xlabel='epochs'
        )
        opts['ylabel'] = 'loss'
        loss_plot = vis.line(np.array([initial_loss]), np.array([0]),
            opts=opts)
        opts['ylabel'] = 'success'
        reward_plot = vis.line(np.array([0]), np.array([0]),
            opts=opts)
        rewards_futures = []
        epochs = 1000000
        eval_period = 100
        print('initial loss=%f' % initial_loss)
        cur_future = 0
        for e in range(epochs):
            start_time = time()
            pairs = opt()
            speed = pairs / (time() - start_time)
            if e % eval_period == 0:
                current_loss = compute_loss()
                vis.line(
                    np.array([current_loss]), np.array([e]),
                    win=loss_plot, update='append')
                print('epoch %i: \tloss=%.4f \tspeed=%.1f' % (e, current_loss, speed))
                th.save(policy, 'tmp/models/imitation_th_epoch_%i.pt' % e)
                rewards_futures.append(executor.submit(estimate_reward, e, copy.deepcopy(policy)))
            if len(rewards_futures) > cur_future:
                future = rewards_futures[cur_future]
                if future.done():
                    vis.line(
                        np.array([future.result()]), np.array([cur_future * eval_period]),
                        win=reward_plot, update='append')
                    cur_future += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action='store')
    params = {'lr': 1e-6, 'batch': 32, 'decay': 1e-2}
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))
    args = parser.parse_args()

    with h5py.File('tmp/imitation/' + args.dataset, 'r') as f:
        obs_dataset = np.array(f['obsVar'])
        act_dataset = np.array(f['actVar'])

    n = obs_dataset.shape[0] * obs_dataset.shape[1]
    obs_dim = obs_dataset.shape[2]
    act_dim = act_dataset.shape[2]
    w, h, c = size(obs_dim, 2, 1)
    print('w = %i, h = %i, c = %i' % (w, h, c))
    train(
        th.from_numpy(obs_dataset.reshape((n, h, w, c)).transpose(0, 3, 1, 2)).cuda(),
        th.from_numpy(act_dataset.reshape((n, act_dim))).cuda(),
        Policy(w, h, c, act_dim),
        args
    )

def play(args, env):
    epoch = 5
    w, h, c = size(env.observation_dim, 2, 1)
    policy = th.load('tmp/models/imitation_th_epoch_%i.pt' % epoch)
    n_eps = 50
    def act(obs):
        with th.no_grad():
            obs_th = th.from_numpy(obs.reshape((1, h, w, c)).transpose(0, 3, 1, 2)).float().cuda()
            _, action, _, _ = policy.act(obs_th, deterministic=True)
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

def estimate_reward(epoch, policy):
    try:
        print('estimating %i...' % epoch)
        env = minecraft.environment.MinecraftEnv('GoBreakGold')
        env.init_spaces()
        n_eps = 50
        w, h, c = size(env.observation_dim, 2, 1)
        def act(obs):
            with th.no_grad():
                obs_th = th.from_numpy(obs.reshape((1, h, w, c)).transpose(0, 3, 1, 2)).float().cuda()
                _, action, _, _ = policy.act(obs_th, deterministic=True)
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
    main()
