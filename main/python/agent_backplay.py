import argparse
from minecraft.environment import *
import random
import subprocess
import time
import threading
import sys
import torch as th
import h5py
import agent_imitation_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('partial_command', action='store')
    params = {'t_offset': 10, 't_spread': 10}
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))
    args = parser.parse_args()

    agents_ready = False
    steps_elapsed = 0
    steps_limit = 60
    loaded_t = None

    env_type = 'Pattern'
    record = 'part1'
    trajs = read_trajs(record)

    traj_obs, traj_start, traj_end = trajs[0]

    def states_initializer(policy, actor_id):
        with th.no_grad():
            _, _, _, states = policy.act(traj_obs[actor_id, traj_start:loaded_t])
        return states

    def step(done):
        nonlocal steps_elapsed, steps_limit, loaded_t
        if not agents_ready:
            return (MinecraftController.WAIT, None)
        if not done:
            steps_elapsed += 1
            if steps_elapsed >= steps_limit:
                done = True
                steps_elapsed = 0
        if done:
            offset = random.randint(args.t_offset - args.t_spread, args.t_offset + args.t_spread + 1)
            steps_limit = 2 * offset
            loaded_t = traj_end - offset
            return (MinecraftController.LOAD, loaded_t)
        else:
            return (MinecraftController.WAIT, None)

    c = MinecraftController(env_type, step, record)
    c.start()

    command = "%s --env-name mc.%s.%s" % (args.partial_command, c.env_type, c.env_id)
    sys.argv = command.split()

    def rl_main():
        import rl.main
        rl.main.main(states_initializer=states_initializer)

    thread = threading.Thread(target=rl_main)
    thread.start()

    time.sleep(2)
    agents_ready = True

    thread.join()

def read_trajs(filename):
    trajs = []
    with h5py.File('tmp/imitation/' + filename + '.h5', 'r') as f:
        obs = np.array(f['obsVar'])
        act = np.array(f['actVar'])
        env_info = np.array(f['envInfoVar'])
        n_samples = obs.shape[0]
        n_humans = obs.shape[1]
        obs_dim = obs.shape[2]
        act_dim = act.shape[2]
        w, h, c = agent_imitation_train.size(obs_dim, 2)
        print('w = %i, h = %i, c = %i' % (w, h, c))
        traj_start = 0
        while traj_start < n_samples:
            traj_end = traj_start
            while traj_end < n_samples and env_info[traj_end, 0, 1] == 0:
                traj_end += 1
            if traj_end > traj_start and traj_end < n_samples:
                n = traj_end+1 - traj_start
                trajs.append((
                    th.from_numpy(
                        obs[traj_start:traj_end+1, :, :].reshape((n, n_humans, h, w, c)).transpose(1, 0, 4, 2, 3)
                    ).cuda(),
                    traj_start,
                    traj_end,
                ))
            traj_start = traj_end + 1
    return trajs

if __name__ == '__main__':
    main()
