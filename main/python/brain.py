from __future__ import print_function
import sys

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)
	sys.stderr.flush()
	
class State:
	def __init__(self):
		self.x = 0
		self.y = 0
		self.z = 0

		self.yaw = 0
		self.pitch = 0
		self.blocks = [None] * 27

		self.forward = 0
		self.strafe = 0
		self.momentumYaw = 0
		self.momentumPitch = 0

		self.jump = False
		self.crouch = False
		self.attack = False
		self.use = False

def act(state):
	print(state.forward)
	print(state.strafe)
	print(state.momentumYaw)
	print(state.momentumPitch)
	print(state.jump)
	print(state.crouch)
	print(state.attack)
	print(state.use)
	sys.stdout.flush()
	
def observe(state):
	state.x = float(sys.stdin.readline())
	state.y = float(sys.stdin.readline())
	state.z = float(sys.stdin.readline())
	state.yaw = float(sys.stdin.readline())
	state.pitch = float(sys.stdin.readline())
	for i in range(len(state.blocks)):
		state.blocks[i] = sys.stdin.readline()

s = State()
		
while True:
	act(s)
	observe(s)
	