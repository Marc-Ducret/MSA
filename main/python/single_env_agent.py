from minecraft_environment import *
import argparse
import time
import baselines.logger as logger

def _run(args, run):
    env = MinecraftEnv(args.env_type, args.env_id)
    env.init_spaces()
    run(args, env)
    env.close()

def run_agent(run, params={}):
    logger.Logger.CURRENT = logger.Logger('tmp/logs/', [
        logger.TensorBoardOutputFormat('tmp/logs/run_%i' % int(time.time() * 1e2)),
        logger.HumanOutputFormat(sys.stdout)])
    parser = argparse.ArgumentParser()
    parser.add_argument('env_type', action='store')
    params.update({'env_id': ""})
    for par, default in params.items():
        parser.add_argument('--'+par, action='store', default=default, type=type(default))

    _run(parser.parse_args(), run)
