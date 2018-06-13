import subprocess


for R in [2, 4, 6, 8, 10]:
    commands = []
    for C in [2, 3, 4]:
        cmd = ('python python/agent_rl.py --env-name mc.Trade[%i,%i,100].trade[%i,%i] --num-processes %i --no-vis --num-stack 1 ' +
            '--algo ppo --num-steps 500 --entropy-coef 0 --ppo-epoch 3 --lr 1e-3 --num-frames 200000') % (C, R, C, R, 1)
        for _ in range(C):
            commands.append(cmd)
    processes = [subprocess.Popen(command.split()) for command in commands]

    for p in processes:
        p.wait()
