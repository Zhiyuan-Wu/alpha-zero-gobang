import logging

import coloredlogs

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
    'game_size': 15,                         # Board size
    'numIters': 100,                         # Not used
    'episode_size': 100000,                  # Number of samples to simulate during a new iteration.
    'tempThreshold': 5,                      # MCTS result policy tempreture will reduce 0 after this number of game turns
    'updateThreshold': 0.55,                 # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'numMCTSSims': 1600,                     # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,                      # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,                              # Higher value lead to less exploration
    'endGameRewardWeight': 1,                # Amplify the real endgame reward over network estimation
    'dirichlet_alpha': 0.03,                 # Parameter alpha of dirichlet noise
    'dirichlet_weight': 0.3,                 # Parameter epsilon of dirichlet noise
    'softmax_temp': 1.0,
    'keep_mcts_after_move': False,
             
    'sampler_num': 32,                       # The number of parallel sampling process.
    'mcts_per_sampler': 64,                  # The number of mcts inside each sampling process, higher value lead to larger query batch
    'gpu_evaluator': [4,5,6,7],              # The GPU used as evaluator
    'gpu_trainner':  [0,1,2],                # The GPU used as trainner
    'gpu_arena': 3,                          # The GPU used as arena
    'available_mem_gb': 100,                 # A number that limit the usage of memory
    'tqdm_wait_time': 0.1,                   # timeout parameter for global lock. tqdm randomly deadlock without this.
    'port': 47152,                           # localhost port number for pytorch ddp

    'checkpoint': './result0816/',           # checkpoint saving directory
    'load_model': True,                      # load a checkpoint to start
    'model_series_number': 1694505322,       # the load model series number
    'numItersForTrainExamplesHistory': 30,   # the maximum iterations that sample buffer keeps
    'leastTrainingWindow': 29,               # the lowest number of iterations to start training

    'lr': 0.001,                             # learning rate
    'pi_loss_weight': 1.0,                   # loss function weight for policy term
    'dropout': 0.3,                          # dropout rate
    'epochs': 1,                             # number of training epochs for each iteration
    'batch_size': 512,                       # trainning batchsize
    'cuda': torch.cuda.is_available(),       # if cuda avaliable. Note: not tested when cuda not avaliable
    'num_channels': 128,                     # neural network feature channel number
    'block_num': 6,                          # neural network residual convolution block number
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
