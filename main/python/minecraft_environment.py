from __future__ import print_function
import sys
from data_stream import *

from gym import spaces

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

class MinecraftEnv:

    def __init__(self):
        self.in_stream  = DataInputStream (sys.stdin .buffer)
        self.out_stream = DataOutputStream(sys.stdout.buffer)

        self.state_dim = self.in_stream.read_int()
        self.state_space = states.Box(low=-1, high=1, shape=(state_dim,))
        self.action_dim = self.in_stream.read_int()
        self.action_space = states.Box(low=-1, high=1, shape=(action_dim,))

    def _receive_states(self):
        return self.in_stream.read_float_array(self.state_dim)

    def _receive_rewards(self):
        return self.in_stream.read_float()

    def _receive_done(self):
        return self.in_stream.read_boolean()

    def _send_actions(self, action):
        self.out_stream.write_float_array(action)

    def step(self, actions):
        self._send_actions(actions)
        return  self._receive_state(),
                self._receive_reward(),
                self._receive_done(),
                {}

    def reset(self):
        return self._receive_states()
