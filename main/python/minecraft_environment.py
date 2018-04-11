import sys
from data_stream import *
import socket

import numpy as np
import gym
from gym import spaces

class MinecraftEnv(gym.Env):

    def __init__(self, env_type, env_id=None):
        self.env_type = env_type
        self.env_id = env_id
        sok = socket.create_connection(('localhost', 1337))
        sok.settimeout(2)
        self.in_stream  = DataInputStream (sok.makefile(mode='rb'))
        self.out_stream = DataOutputStream(sok.makefile(mode='wb'))

        self.out_stream.write_utf(env_type)
        if env_id is None:
            self.out_stream.write_boolean(True)
        else:
            self.out_stream.write_boolean(False)
            self.out_stream.write_utf(env_id)
        self.out_stream.flush()

        self.observation_dim = self.in_stream.read_int()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,), dtype=np.float32)
        self.action_dim = self.in_stream.read_int()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

        self.reward_range = (-100, 100) ## TODO: get from java

        self.num_envs = 1

    def params(self):
        return {}

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
        action = action.reshape((self.action_dim,))
        self._send_action(action)
        return  self._receive_observation(), self._receive_reward(), self._receive_done(), {}

    def reset(self):
        self.out_stream.write_int(0x13371337)
        self.out_stream.flush()
        return self._receive_observation()
