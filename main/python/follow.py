from brain import *
from math import *
import re

def think(brain):
	pattern = re.compile(r"""(?P<type>.*?)\[
							'(?P<name>.*?)'/
							(?P<id>.*?),\sl='New\sWorld',\s
							x=(?P<x>.*?),\s
							y=(?P<y>.*?),\s
							z=(?P<z>.*?)\]""", re.VERBOSE)
	for e in brain.state.entities:
		match = pattern.match(e)
		if match is not None:
			type = match.group("type")
			name = match.group("name")
			x = float(match.group("x"))
			y = float(match.group("y"))
			z = float(match.group("z"))
			
			brain.state.forward = z - brain.state.z
			brain.state.strafe = x - brain.state.x
			return
		
	
Brain(1, 10, think).run()