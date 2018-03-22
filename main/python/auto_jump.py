import brain

def think(brain):
	brain.state.forward = 1
	brain.state.jump = brain.state.block(1, -1, 0) != "minecraft:air"

brain.Brain(5, 1, think).run()