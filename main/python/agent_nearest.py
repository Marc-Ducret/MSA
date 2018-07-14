from agent_imitation_train import *
import argparse
import torch as th
import single_env_agent
import h5py

def play(args):
    with th.cuda.device(args.gpu):
        try:
            env = minecraft.environment.MinecraftEnv(args.env_type, args.env_id)
            env.init_spaces()
            n_eps = 50

            skip = 6
            stack = 4
            agent = 0

            with h5py.File('tmp/imitation/' + args.dataset + '.h5', 'r') as f:
                obs_dataset = np.array(f['obsVar'])
                n = obs_dataset.shape[0] * obs_dataset.shape[1]
                n = obs_dataset.shape[0]
                obs_dataset = th.from_numpy(obs_dataset[:,agent,:].reshape((n, env.observation_dim))).float().cuda()
                act_dataset = th.from_numpy(np.array(f['actVar'])[:,agent,:].reshape((n, env.action_dim))).float().cuda()

            def ids_offset(offset):
                return th.clamp(th.arange(0, n) - offset, 0, n).long().cuda()
            obs_data_seq = th.cat([obs_dataset[ids_offset((skip+1) * i)] for i in range(stack)], 1)

            def act(obs, seq_mod, obs_seq):
                with th.no_grad():
                    obs_th = th.from_numpy(obs.reshape((1, env.observation_dim))).float().cuda()
                    if seq_mod < 0:
                        obs_seq = th.cat([obs_th] * stack, 1)
                    else:
                        obs_seq[:, seq_mod * env.observation_dim : (seq_mod+1) * env.observation_dim] = obs_th
                    obs_seq_view = obs_seq.view(1, stack, -1)[:, (th.arange(0, stack).long().cuda() - seq_mod) % stack, :].view(1, -1)
                    dists, ids = th.topk(th.pow(th.mean(obs_data_seq - obs_seq_view, 1), 2), args.k, 0, False)
                    #coef = th.nn.functional.softmax(-dists * 1e0).view(args.k, 1)
                    action = th.mean(act_dataset[ids], 0).cpu().detach().numpy()
                return action, (seq_mod + 1) % stack, obs_seq
            mean_rew = 0
            for i in range(n_eps):
                obs = env.reset()
                ep_rew = 0
                obs_seq = None
                seq_mod = -1
                while True:
                    action, seq_mod, obs_seq = act(obs, seq_mod, obs_seq)
                    obs, rew, done, _ = env.step(action)
                    ep_rew += rew
                    if done:
                        ep_rew = 100 if ep_rew > 0 else 0
                        mean_rew += ep_rew / n_eps
                        break
            print('estimated %.2f' % mean_rew)
            return mean_rew
        except:
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action='store')
    parser.add_argument('k', action='store', type=type(1))
    parser.add_argument('env_type', action='store')
    params = {'gpu': 0, 'env_id': ''}
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))
    args = parser.parse_args()
    play(args)

if __name__ == '__main__':
    main()
