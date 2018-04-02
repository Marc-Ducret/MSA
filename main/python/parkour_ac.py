import utils
from ac_agent import *
from brain import *
from math import *
import re
import numpy as np
import cProfile

def encode(b):
    if b > 0:
        return 1
    else:
        return 0

def think(brain):
    if brain.state.y > 9.5:
        agent.done = False
    if not agent.done:
        s = SIGHT * 2 + 1
        dz = -(brain.state.z + 566) / 22
        rel_x = brain.state.x - floor(brain.state.x)
        rel_z = brain.state.z - floor(brain.state.z)
        state = np.array([encode(brain.state.block(i % s - SIGHT, -1, i // s - SIGHT)) for i in range(s ** 2)])
        state = np.append(state, np.array([dz, rel_x, rel_z, 1])).reshape((1, agent.state_dim))

        reward = (0 - dz) * .005

        if brain.state.y < 9:
            agent.done = True
            reward -= 100 * dz

        agent.score += reward

        if agent.prev_state is not None:
            agent.remember(agent.prev_state, agent.action, reward, state, agent.done)

        agent.action = agent.act(state).reshape((1, agent.action_dim))
        brain.state.forward = agent.action[0][0]
        brain.state.strafe = agent.action[0][1]
        agent.prev_state = state

        if agent.done:
            eprint('restart | score=', agent.score)
            agent.score = 0

    if brain.state.age % 6 == 0:
        agent.train()
        if brain.state.age % 60 == 0:
            agent.update_target()

sess = utils.use_gpu(False)
SIGHT = 1

agent = ActorCritic((SIGHT * 2 + 1) ** 2 + 4, 2, sess)
agent.lr_decay = .999
agent.epsilon_decay = .997
agent.gamma = 1
agent.prev_state = None
agent.done = False
agent.score = 0

batch_size = 32

pattern = re.compile(r"""(?P<type>.*?)\[
                        '(?P<name>.*?)'/
                        (?P<id>.*?),\sl='(?P<world_name>.*?)',\s
                        x=(?P<x>.*?),\s
                        y=(?P<y>.*?),\s
                        z=(?P<z>.*?)\]""", re.VERBOSE)

Brain(1, 5, think).run()
