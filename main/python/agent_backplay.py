import argparse
from minecraft.environment import *
import random
import subprocess
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('partial_command', action='store')
    params = {'t_min': 0, 't_max': 1000 }
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))
    args = parser.parse_args()

    agents_ready = False

    def step(done):
        return (MinecraftController.LOAD, random.randint(args.t_min, args.t_max)) if done and agents_ready else (MinecraftController.WAIT, None)

    c = MinecraftController("Pattern", step, "part1")
    c.start()

    command = "%s --env-name mc.%s.%s" % (args.partial_command, c.env_type, c.env_id)
    processes = [subprocess.Popen(command.split()) for _ in range(1)]

    time.sleep(2)
    agents_ready = True

    for p in processes:
        p.wait()

if __name__ == '__main__':
    main()
