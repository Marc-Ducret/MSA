from __future__ import print_function
import sys
from data_stream import *

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

class State:
    def __init__(self, obs_dist):
        self.obs_dist = obs_dist

        self.x = 0
        self.y = 0
        self.z = 0

        self.yaw = 0
        self.pitch = 0
        self.blocks = [None] * ((obs_dist * 2 + 1) ** 3)
        self.entities = []

        self.forward = 0
        self.strafe = 0
        self.momentum_yaw = 0
        self.momentum_pitch = 0

        self.jump = False
        self.crouch = False
        self.attack = False
        self.use = False

    def block(self, dx, dy, dz):
        if max(abs(dx), abs(dy), abs(dz)) > self.obs_dist:
            raise str(dx)+" "+str(dy)+" "+str(dz)+" is too far from agent (max dist: "+str(self.obs_dist)+")"
        d = self.obs_dist
        dx += d
        dy += d
        dz += d
        d = d * 2 + 1
        return self.blocks[dx + dy * d + dz * d * d]

class Brain:
    def __init__(self, update_period, obs_dist, think):
        self.update_period = update_period
        self.obs_dist = obs_dist
        self.state = State(obs_dist)
        self.think = think
        self.inStream = DataInputStream(sys.stdin.buffer)
        self.outStream = DataOutputStream(sys.stdout.buffer)

    def init(self):
        self.outStream.write_int(self.update_period)
        self.outStream.write_int(self.obs_dist)

    def act(self):
        self.outStream.write_float(self.state.forward)
        self.outStream.write_float(self.state.strafe)
        self.outStream.write_float(self.state.momentum_yaw)
        self.outStream.write_float(self.state.momentum_pitch)
        self.outStream.write_boolean(self.state.jump)
        self.outStream.write_boolean(self.state.crouch)
        self.outStream.write_boolean(self.state.attack)
        self.outStream.write_boolean(self.state.use)
        self.outStream.flush()

    def observe(self):
        def read():
            return sys.stdin.readline()[:-1]
        self.state.x = self.inStream.read_float()
        self.state.y = self.inStream.read_float()
        self.state.z = self.inStream.read_float()
        self.state.yaw = self.inStream.read_float()
        self.state.pitch = self.inStream.read_float()
        for i in range(len(self.state.blocks)):
            self.state.blocks[i] = self.inStream.read_int()
        self.state.entities = ["#"] * self.inStream.read_int()
        for i in range(len(self.state.entities)):
            self.state.entities[i] = self.inStream.read_utf()

    def run(self):
        self.init()
        while True:
            self.act()
            self.observe()
            if self.think(self) is not None:
                return
