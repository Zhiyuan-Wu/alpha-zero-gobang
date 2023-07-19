import logging

import coloredlogs

from Coach import Coach
from newCoach import mpCoach
from gobang.GobangGame import GobangGame as Game
from gobang.pytorch.NNet import NNetWrapper as nn
from utils import *
import multiprocessing as mp
if __name__ =="__main__":
    mp.set_start_method('spawn')

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 100,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 1000,        # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,                 # Higher value lead to more exploration
    'endGameRewardWeight': 1,   # Amplify the real endgame reward over network estimation

    'sampler_num': 32,          # The number of parallel sampling process.
    'mcts_per_sampler': 16,     # The number of mcts inside each sampling process, higher value lead to larger query batch, but not overall faster sampling
    'gpu_num': 4,               # The number of avaliable gpu, gpu:0 will be used to train model, others will be used to inference
    'available_mem_gb': 200,
    'tqdm_wait_time': 0.1,      # timeout parameter for global lock. tqdm randomly deadlock without this.

    'checkpoint': './result0719/',
    'load_model': True,
    'load_folder_file': ('./result0719/','checkpoint_1689767719.pth.tar'),
    'numItersForTrainExamplesHistory': 30,
})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(15)

    # if args.load_model:
    #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # else:
    #     log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    # c = Coach(game=g, nnet=nn(g), args=args)
    c = mpCoach(game=g, nnet=nn, args=args)

    # if args.load_model:
    #     log.info("Loading 'trainExamples' from file...")
    #     c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
