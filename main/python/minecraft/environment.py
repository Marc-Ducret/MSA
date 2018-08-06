import sys
from .data_stream import *
import socket
import threading

import numpy as np
import gym
from gym import spaces

class MinecraftEnv(gym.Env):

    def __init__(self, env_type, env_id='', use_entities=False):
        self.env_type = env_type
        self.env_id = env_id
        self.use_entities = use_entities

        if env_id == '':
            controller = MinecraftController(env_type, MinecraftController.default_step)
            controller.start()
            env_id = controller.env_id
            self.env_id = env_id

        self.sok = socket.create_connection(('localhost', 1337))
        self.sok.settimeout(5)
        self.in_stream  = DataInputStream (self.sok.makefile(mode='rb'))
        self.out_stream = DataOutputStream(self.sok.makefile(mode='wb'))

        self.out_stream.write_utf(env_type)
        self.out_stream.write_boolean(False)
        self.out_stream.write_utf(env_id)
        self.out_stream.flush()

        self.num_envs = 1

        self.ep_len = 0
        self.ep_rew = 0

    def init_spaces(self):
        self.env_id = self.in_stream.read_utf()
        self.observation_dim = self.in_stream.read_int()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,), dtype=np.float32)
        self.action_dim = self.in_stream.read_int()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.entity_dim = self.in_stream.read_int()
        self.entity_max = self.in_stream.read_int()
        self.actor_id = self.in_stream.read_int()

        self.reward_range = (-100, 100) ## TODO: get from java

    def params(self):
        return {}

    def _receive_observation(self):
        obs = np.array(self.in_stream.read_float_array(self.observation_dim))
        if self.entity_max > 0:
            size = self.in_stream.read_int()
            obs_entities = np.array(self.in_stream.read_float_array(size))
            obs_entities.resize((self.entity_max * self.entity_dim,))
            entity_mask = np.array([i < size // self.entity_dim for i in range(self.entity_max)])
            return (obs, obs_entities, entity_mask) if self.use_entities else obs
        else:
            return (obs, np.zeros((0,)), np.zeros((0,))) if self.use_entities else obs

    def _receive_reward(self):
        return self.in_stream.read_float()

    def _receive_done(self):
        return self.in_stream.read_boolean()

    def _send_action(self, action):
        self.out_stream.write_float_array(action)
        self.out_stream.flush()

    def step(self, action):
        self.step_act(action)
        return self.step_result()

    def step_act(self, action):
        action = action.reshape((self.action_dim,))
        self._send_action(action)

    def step_result(self):
        obs, rew, done, info = self._receive_observation(), self._receive_reward(), self._receive_done(), {}
        self.ep_len += 1
        self.ep_rew += rew
        if done:
            info.update({'episode': {'l': self.ep_len, 'r': self.ep_rew}})
            self.ep_len = 0
            self.ep_rew = 0
        return obs, rew, done, info

    def reset(self):
        return self._receive_observation()

    def close(self):
        self.sok.close()

class MinecraftController:
    WAIT = 0
    RESET = 1
    LOAD = 2
    TERMINATE = 3

    def default_step(done):
        return (MinecraftController.RESET if done else MinecraftController.WAIT), None

    def __init__(self, env_type, step, record=None):
        self.env_type = env_type
        self.sok = socket.create_connection(('localhost', 1337))
        self.sok.settimeout(5)
        self.in_stream  = DataInputStream (self.sok.makefile(mode='rb'))
        self.out_stream = DataOutputStream(self.sok.makefile(mode='wb'))

        self.out_stream.write_utf(env_type)
        self.out_stream.write_boolean(True)
        self.out_stream.write_boolean(record is not None)
        if record is not None:
            self.out_stream.write_utf(record)
        self.out_stream.flush()

        self.step = step

    def start(self):
        self.env_id = self.in_stream.read_utf()

        self.thread = threading.Thread(target=self._socket_thread)
        self.thread.deamon = True
        self.thread.start()

    def _socket_thread(self):
        while True:
            done = self.in_stream.read_boolean()
            state, param = self.step(done)
            self.out_stream.write_byte(state)
            if param is not None:
                self.out_stream.write_int(param)
            self.out_stream.flush()
