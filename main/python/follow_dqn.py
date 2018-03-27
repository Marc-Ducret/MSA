from dqnagent import *
from brain import *
from math import *
import re
import numpy as np
import cProfile

agent = DQNAgent(2, 4)
agent.prev_state = None

batch_size = 32

pattern = re.compile(r"""(?P<type>.*?)\[
							'(?P<name>.*?)'/
							(?P<id>.*?),\sl='New\sWorld',\s
							x=(?P<x>.*?),\s
							y=(?P<y>.*?),\s
							z=(?P<z>.*?)\]""", re.VERBOSE)

							
def think(brain):					
	dx, dz = None, None
							
	for e in brain.state.entities:
		match = pattern.match(e)
		if match is not None:
			type = match.group("type")
			name = match.group("name")
			x = float(match.group("x"))
			y = float(match.group("y"))
			z = float(match.group("z"))
			
			dz = z - brain.state.z
			dx = x - brain.state.x
			break
	
	if dx is not None:
		dx /= 5.0
		dz /= 5.0
		reward = - (dx ** 2 + dz ** 2)
		eprint(reward)
		state = np.array([dx, dz])
		state = np.reshape(state, [1, 2])
		if agent.prev_state is not None:
			agent.remember(agent.prev_state, agent.action, reward, state, False)

		agent.action = agent.act(state)
		brain.state.forward = cos(agent.action * pi / 2) * .5
		brain.state.strafe = sin(agent.action * pi / 2) * .5
		agent.prev_state = state
		
	if len(agent.memory) > batch_size:
		agent.replay(batch_size)

Brain(1, 5, think).run()