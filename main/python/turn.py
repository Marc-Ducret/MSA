from brain import *
from math import *

def think(brain):
	yaw = (brain.state.yaw + 90) % 360
	dir = (int((yaw + 45) / 90) * 90) % 360
	

	b = brain.state.block(int(cos(radians(dir))), 0, int(sin(radians(dir))))
	
	brain.state.forward = 0
	brain.state.momentum_yaw = 0
	
	if b > 0:
			brain.state.momentum_yaw = -.2
	else:	
		if abs(dir - yaw) > 1:
			brain.state.momentum_yaw = (dir - yaw) / 45
		else:
			brain.state.forward = 1
		
	
Brain(1, 1, think).run()