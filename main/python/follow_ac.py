import utils
from ac_agent import *
from brain import *
from math import *
import re
import numpy as np
import cProfile

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
        brain.state.crouch = False
        dx /= 5.0
        dz /= 5.0
        reward = - (abs(dx) ** 2 + abs(dz) ** 2)
        state = np.array([dx, dz]).reshape((1, agent.state_dim))
        if agent.prev_state is not None:
            agent.remember(agent.prev_state, agent.action, reward, state, False)

        agent.action = agent.act(state).reshape((1, agent.action_dim))
        brain.state.forward = agent.action[0][0]
        brain.state.strafe = agent.action[0][1]
        agent.prev_state = state

        if brain.state.age % 6 == 0:
            agent.train()
            if brain.state.age % 60 == 0:
                agent.update_target()
    else:
        brain.state.forward = 0
        brain.state.strafe = 0
        brain.state.crouch = True

sess = utils.use_gpu(False)

agent = ActorCritic(2, 2, sess)
agent.prev_state = None

batch_size = 32

pattern = re.compile(r"""(?P<type>.*?)\[
                        '(?P<name>.*?)'/
                        (?P<id>.*?),\sl='(?P<world_name>.*?)',\s
                        x=(?P<x>.*?),\s
                        y=(?P<y>.*?),\s
                        z=(?P<z>.*?)\]""", re.VERBOSE)

Brain(1, 5, think).run()
