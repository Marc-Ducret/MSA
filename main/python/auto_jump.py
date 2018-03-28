from brain import *

def think(brain):
    brain.state.forward = .5
    brain.state.jump = brain.state.block(0, 0, 1) != "minecraft:air"

Brain(5, 1, think).run()
