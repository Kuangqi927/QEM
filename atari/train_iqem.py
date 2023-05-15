import os
import yaml
import argparse
import torch
from datetime import datetime

from iqn_qrdqn.env import make_pytorch_env
from iqn_qrdqn.agent import IQEMAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    env = make_pytorch_env(args.env_id)
    test_env = make_pytorch_env(
        args.env_id, episode_life=False, clip_rewards=False)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', 'IQEM', args.env_id, f'{name}-seed{args.seed}-{time}')
    
    torch.cuda.set_device(args.cuda)
    
    # Create the agent and run.
    agent = IQEMAgent(
        env=env, test_env=test_env, log_dir=log_dir,explo=args.explo,algo=args.algo,seed=args.seed,**config)

    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'iqem.yaml'))
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', type=int, default=0)
    #parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--explo', type=str, default="none")
    parser.add_argument('--algo', type=str, default="qem")
    
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    run(args)
