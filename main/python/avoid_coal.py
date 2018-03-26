from dqnagent import *
from brain import *
from math import *
import re
import numpy as np
import cProfile

SIGHT = 1

agent = DQNAgent((SIGHT * 2 + 1) ** 2, 4, hidden=32)
agent.prev_state = None

batch_size = 32
mean_reward = 0

def encode(b):
	if b == 'minecraft:coal_block':
		return -1
	if b == 'minecraft:gold_block':
		return 1
	return 0

def think(brain):
	reward = encode(brain.state.block(0, -1, 0))
	global mean_reward
	mean_reward = mean_reward * .999 + reward * .001
	eprint('cur:', reward, 'mean:', mean_reward)
	s = SIGHT * 2 + 1
	state = np.array([encode(brain.state.block(i % s - SIGHT, -1, i // s - SIGHT)) for i in range(s ** 2)])
	state = np.reshape(state, [1, s ** 2])
	eprint(state)
	if agent.prev_state is not None:
		agent.remember(agent.prev_state, agent.action, reward, state, False)

	agent.action = agent.act(state)
	brain.state.forward = cos(agent.action * pi / 2) * .5
	brain.state.strafe = sin(agent.action * pi / 2) * .5
	agent.prev_state = state
		
	if len(agent.memory) > batch_size:
		agent.replay(batch_size)

Brain(5, SIGHT, think).run()