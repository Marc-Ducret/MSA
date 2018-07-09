from agent_imitation_train import *
import argparse
import torch as th
import single_env_agent

def play(args):
    with th.cuda.device(args.gpu):
        try:
            policy = th.load(args.save, map_location=lambda storage, loc: storage.cuda(args.gpu))
            env = minecraft.environment.MinecraftEnv(args.env_type, args.env_id)
            env.init_spaces()
            n_eps = 50
            w, h, c = size(env.observation_dim, 2)
            def act(obs, states):
                with th.no_grad():
                    obs_th = th.from_numpy(obs.reshape((1, h, w, c)).transpose(0, 3, 1, 2)).float().cuda()
                    _, action, _, states = policy.act(obs_th, states, deterministic=True)
                    action = action.cpu().detach().numpy()
                return action, states
            mean_rew = 0
            for i in range(n_eps):
                obs = env.reset()
                ep_rew = 0
                states = None
                while True:
                    action, states = act(obs, states)
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
    parser.add_argument('save', action='store')
    parser.add_argument('env_type', action='store')
    params = {'gpu': 0, 'env_id': ''}
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))
    args = parser.parse_args()
    play(args)

if __name__ == '__main__':
    main()
