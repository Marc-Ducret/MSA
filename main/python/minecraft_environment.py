from __future__ import print_function
import sys
from data_stream import *

import numpy as np
import gym
from gym import spaces

def _eprint(*args, file=None, **kwargs):
    oldprint(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

class MinecraftEnv(gym.Env):

    def __init__(self):
        self.in_stream  = DataInputStream (sys.stdin .buffer)
        self.out_stream = DataOutputStream(sys.stdout.buffer)

        self.observation_dim = self.in_stream.read_int()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
        self.action_dim = self.in_stream.read_int()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))

        self.reward_range = (-100, 100) ## TODO: get from java

        self.num_envs = 1
        __builtins__['oldprint'] = __builtins__['print']
        __builtins__['print'] = _eprint
        sys.stdout = sys.stderr

    def _receive_observation(self):
        return np.array(self.in_stream.read_float_array(self.observation_dim))

    def _receive_reward(self):
        return self.in_stream.read_float()

    def _receive_done(self):
        return self.in_stream.read_boolean()

    def _send_action(self, action):
        self.out_stream.write_float_array(action)
        self.out_stream.flush()

    def step(self, action):
        action = action.reshape((2,))
        self._send_action(action)
        return  self._receive_observation(), self._receive_reward(), self._receive_done(), {}

    def reset(self):
        self.out_stream.write_int(0x13371337)
        self.out_stream.flush()
        return self._receive_observation()
