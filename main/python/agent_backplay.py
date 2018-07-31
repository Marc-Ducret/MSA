import argparse
from minecraft.environment import *
import random
import subprocess
import time
import threading
import sys
import torch as th

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('partial_command', action='store')
    params = {'t_min': 0, 't_max': 1000 }
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))
    args = parser.parse_args()

    agents_ready = False
    steps_elapsed = 0
    steps_limit = 60

    def step(done):
        if not agents_ready:
            return (MinecraftController.WAIT, None)
        if not done:
            nonlocal steps_elapsed
            steps_elapsed += 1
            if steps_elapsed >= steps_limit:
                done = True
                steps_elapsed = 0
        return (MinecraftController.LOAD, random.randint(args.t_min, args.t_max)) if done else (MinecraftController.WAIT, None)

    c = MinecraftController("Pattern", step, "part1")
    c.start()

    command = "%s --env-name mc.%s.%s" % (args.partial_command, c.env_type, c.env_id)
    sys.argv = command.split()
    thread = threading.Thread(target=rl_main)
    thread.start()

    time.sleep(2)
    agents_ready = True

    thread.join()

def states_initializer(policy, actor_id):
    #print('reset', actor_id)
    return th.zeros(policy.state_size)

def rl_main():
    import rl.main
    rl.main.main(states_initializer=states_initializer)

if __name__ == '__main__':
    main()
