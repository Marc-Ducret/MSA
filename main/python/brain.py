from __future__ import print_function
import sys

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
		
	def init(self):
		print(self.update_period)
		print(self.obs_dist)

	def act(self):
		print(self.state.forward)
		print(self.state.strafe)
		print(self.state.momentum_yaw)
		print(self.state.momentum_pitch)
		print(self.state.jump)
		print(self.state.crouch)
		print(self.state.attack)
		print(self.state.use)
		sys.stdout.flush()
		
	def observe(self):
		def read():
			return sys.stdin.readline()[:-1]
		self.state.x = float(read())
		self.state.y = float(read())
		self.state.z = float(read())
		self.state.yaw = float(read())
		self.state.pitch = float(read())
		for i in range(len(self.state.blocks)):
			self.state.blocks[i] = read()
			
	def run(self):
		self.init()
		while True:
			self.act()
			self.observe()
			self.think(self)