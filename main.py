import logging

import coloredlogs

from Coach import Coach
from newCoach import mpCoach
from gobang.GobangGame import GobangGame as Game
from utils import *
import torch
import torch.multiprocessing as mp
if __name__ =="__main__":
    mp.set_start_method('spawn')

log = logging.getLogger(__name__)

coloredlogs.install(level='ERROR')  # Change this to DEBUG to see more info.

args = dotdict({
    'game_size': 9,                         # Board size
    'numIters': 100,
    'episode_size': 100000,                  # Number of samples to simulate during a new iteration.
    'tempThreshold': 15,                     # MCTS result policy tempreture will reduce 0 after this number of game turns
    'updateThreshold': 0.55,                 # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'numMCTSSims': 1000,                     # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,                      # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,                              # Higher value lead to more exploration
    'endGameRewardWeight': 1,                # Amplify the real endgame reward over network estimation
             
    'sampler_num': 15,                       # The number of parallel sampling process.
    'mcts_per_sampler': 64,                  # The number of mcts inside each sampling process, higher value lead to larger query batch
    'gpu_evaluator': [3,4,5,6,7],            # The GPU used as evaluator
    'gpu_trainner':  [0,1,2],                # The GPU used as trainner
    'available_mem_gb': 150,        
    'tqdm_wait_time': 0.1,                   # timeout parameter for global lock. tqdm randomly deadlock without this.
    'port': 47152,                           # localhost port number for pytorch ddp

    'checkpoint': './result0723/',           # checkpoint saving directory
    'load_model': False,                      # load a check point to start
    'model_series_number': 1690034711,       # the load model series number
    'numItersForTrainExamplesHistory': 30,   # the maximum iterations that sample buffer keeps

    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 1,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'block_num': 6,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(args.game_size)

    log.info('Loading the Coach...')
    c = mpCoach(game=g, args=args)

    log.info('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()
