from minecraft.environment import MinecraftEnv
import argparse
import subprocess

def alloc_env(env_type):
    dummy = MinecraftEnv(env_type)
    dummy.init_spaces()
    dummy.close()
    return dummy.env_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('count', action='store', type=int)
    parser.add_argument('env_type', action='store')
    parser.add_argument('partial_command', action='store')

    args = parser.parse_args()
    env_id = alloc_env(args.env_type)
    command = "%s --env_id %s %s" % (args.partial_command, env_id, args.env_type)

    processes = [subprocess.Popen(command) for _ in range(args.count)]

    for p in processes:
        p.wait()

if __name__ == '__main__':
    main()
