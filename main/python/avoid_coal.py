from dqnagent import *
from brain import *
from math import *
import re
import numpy as np
import cProfile

SIGHT = 1

agent = DQNAgent((SIGHT * 2 + 1) ** 2 + 2, 5, hidden=32)
agent.prev_state = None

batch_size = 32
mean_reward = 0

def encode(b):
    if b == 173: # coal
        return -1
    if b == 41: # gold
        return 1
    return 0

def think(brain):
    rel_x, rel_z = brain.state.x - floor(brain.state.x), brain.state.z - floor(brain.state.z)
    reward = encode(brain.state.block(0, -1, 0)) - .1 * ((rel_x - .5) ** 2 + (rel_z - .5) ** 2)
    global mean_reward
    mean_reward = mean_reward * .99 + reward * .01
    eprint('cur:', reward, 'mean:', mean_reward)
    s = SIGHT * 2 + 1
    state = np.array([encode(brain.state.block(i % s - SIGHT, -1, i // s - SIGHT)) for i in range(s ** 2)])
    state = np.append(state,
                    np.array([rel_x, rel_z]))
    state = np.reshape(state, [1, s ** 2 + 2])
    eprint(state)
    if agent.prev_state is not None:
        agent.remember(agent.prev_state, agent.action, reward, state, False)

    agent.action = agent.act(state)
    if agent.action == 4:
        brain.state.forward = 0
        brain.state.strafe = 0
    else:
        brain.state.forward = cos(agent.action * pi / 2) * .5
        brain.state.strafe = sin(agent.action * pi / 2) * .5
    agent.prev_state = state

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

Brain(1, SIGHT, think).run()
