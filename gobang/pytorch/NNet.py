import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *

import torch
import torch.optim as optim

from .GobangNNet import GobangNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'block_num': 6,
})


class NNetWrapper():
    def __init__(self, game, gpu_id=0, _tqdm=None):
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        if _tqdm is None:
            self.t_bar = tqdm(total=1, desc='Training', position=0)
        else:
            self.t_bar = _tqdm

        if args.cuda:
            self.gpu_nums = torch.cuda.device_count()
            self.gpu_id = gpu_id
            self.nnet.cuda(self.gpu_id)

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        batch_count = int(len(examples) / args.batch_size)
        self.t_bar.reset(total=batch_count*args.epochs)
        for epoch in range(args.epochs):
            # print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            for _ in range(batch_count):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(self.gpu_id), target_pis.contiguous().cuda(self.gpu_id), target_vs.contiguous().cuda(self.gpu_id)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                self.t_bar.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.t_bar.update(1)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda(self.gpu_id)
        board = board.view(-1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy().squeeze(), v.data.cpu().numpy().squeeze()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            # print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            # print("Checkpoint Directory exists! ")
            pass
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', remove_prefix=False):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = 'cuda:'+str(self.gpu_id) if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        if remove_prefix:
            consume_prefix_in_state_dict_if_present(checkpoint, 'module.')
            self.nnet.load_state_dict(checkpoint)
        else:
            self.nnet.load_state_dict(checkpoint['state_dict'])
