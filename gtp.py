''' Set up a GTP engine wrapper over the neural network
'''
import re
import time
import random
import numpy as np
from gobang.pytorch.GobangNNet import GobangNNet as onnet
from gobang.GobangGame import GobangGame as Game
import torch
from utils import *
import os
from MCTS import MCTS
import argparse

args = dotdict({
    'game_size': 9, 
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'block_num': 6,
    'numMCTSSims': 1000,
    'cpuct': 1,
    'endGameRewardWeight': 1,
})

def pre_engine(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.split("#")[0]
    s = s.replace("\t", " ")
    return s

def pre_controller(s):
    s = re.sub("[^\t\n -~]", "", s)
    s = s.replace("\t", " ")
    return s

def gtp_boolean(b):
    return "true" if b else "false"

def gtp_list(l):
    return "\n".join(l)

def gtp_color(color):
    # an arbitrary choice amongst a number of possibilities
    return {BLACK: "B", WHITE: "W"}[color]

def gtp_vertex(vertex):
    if vertex == PASS:
        return "pass"
    elif vertex == RESIGN:
        return "resign"
    else:
        x, y = vertex
        return "{}{}".format("ABCDEFGHJKLMNOPQRSTYVWYZ"[x - 1], y)

def gtp_move(color, vertex):
    return " ".join([gtp_color(color), gtp_vertex(vertex)])

def parse_message(message):
    message = pre_engine(message).strip()
    first, rest = (message.split(" ", 1) + [None])[:2]
    if first.isdigit():
        message_id = int(first)
        if rest is not None:
            command, arguments = (rest.split(" ", 1) + [None])[:2]
        else:
            command, arguments = None, None
    else:
        message_id = None
        command, arguments = first, rest

    return message_id, command, arguments

WHITE = +1
BLACK = -1
EMPTY = 0

PASS = (0, 0)
RESIGN = "resign"

def parse_color(color):
    if color.lower() in ["b", "black"]:
        return BLACK
    elif color.lower() in ["w", "white"]:
        return WHITE
    else:
        return False

def parse_vertex(vertex_string):
    if vertex_string is None:
        return False
    elif vertex_string.lower() == "pass":
        return PASS
    elif len(vertex_string) > 1:
        x = "abcdefghjklmnopqrstuvwxyz".find(vertex_string[0].lower()) + 1
        if x == 0:
            return False
        if vertex_string[1:].isdigit():
            y = int(vertex_string[1:])
        else:
            return False
    else:
        return False
    return (x, y)

def parse_move(move_string):
    color_string, vertex_string = (move_string.split(" ") + [None])[:2]
    color = parse_color(color_string)
    if color is False:
        return False
    vertex = parse_vertex(vertex_string)
    if vertex is False:
        return False

    return color, vertex

MIN_BOARD_SIZE = 7
MAX_BOARD_SIZE = 19

def format_success(message_id, response=None):
    if response is None:
        response = ""
    else:
        response = " {}".format(response)
    if message_id:
        return "={}{}\n\n".format(message_id, response)
    else:
        return "={}\n\n".format(response)

def format_error(message_id, response):
    if response:
        response = " {}".format(response)
    if message_id:
        return "?{}{}\n\n".format(message_id, response)
    else:
        return "?{}\n\n".format(response)

class Engine(object):
    def __init__(self, game_obj, name="gtp (python library)", version="0.2"):

        self.size = 19
        self.komi = 6.5

        self._game = game_obj
        self._game.clear()

        self._name = name
        self._version = version

        self.disconnect = False

        self.known_commands = [
            field[4:] for field in dir(self) if field.startswith("cmd_")]

    def send(self, message):
        message_id, command, arguments = parse_message(message)
        if command in self.known_commands:
            try:
                return format_success(
                    message_id, getattr(self, "cmd_" + command)(arguments))
            except ValueError as exception:
                return format_error(message_id, exception.args[0])
        else:
            return format_error(message_id, "unknown command")

    def vertex_in_range(self, vertex):
        if vertex == PASS:
            return True
        if 1 <= vertex[0] <= self.size and 1 <= vertex[1] <= self.size:
            return True
        else:
            return False

    # commands

    def cmd_protocol_version(self, arguments):
        return 2

    def cmd_name(self, arguments):
        return self._name

    def cmd_version(self, arguments):
        return self._version

    def cmd_known_command(self, arguments):
        return gtp_boolean(arguments in self.known_commands)

    def cmd_list_commands(self, arguments):
        return gtp_list(self.known_commands)

    def cmd_quit(self, arguments):
        self.disconnect = True
        exit()

    def cmd_boardsize(self, arguments):
        if arguments.isdigit():
            size = int(arguments)
            if MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE:
                self.size = size
                self._game.set_size(size)
            else:
                raise ValueError("unacceptable size")
        else:
            raise ValueError("non digit size")

    def cmd_clear_board(self, arguments):
        self._game.clear()

    def cmd_komi(self, arguments):
        try:
            komi = float(arguments)
            self.komi = komi
            self._game.set_komi(komi)
        except ValueError:
            raise ValueError("syntax error")

    def cmd_play(self, arguments):
        move = parse_move(arguments)
        if move:
            color, vertex = move
            if self.vertex_in_range(vertex):
                if self._game.make_move(color, vertex):
                    return
        raise ValueError("illegal move")

    def cmd_genmove(self, arguments):
        c = parse_color(arguments)
        if c:
            move = self._game.get_move(c)
            self._game.make_move(c, move)
            return gtp_vertex(move)
        else:
            raise ValueError("unknown player: {}".format(arguments))

class NeuralPlayer():
    def __init__(self, size=19, komi=6.5):
        self.size = size
        self.komi = komi
        self.board = np.zeros((size, size), dtype=np.int8)
        self.g = Game(size)
        self.nnet = onnet(self.g, args)
        _path = args.model_path
        if args.cuda:
            map_location = 'cuda:0'
            self.nnet.cuda(0)
        else:
            map_location = 'cpu'
        state_dict = torch.load(_path, map_location=map_location)
        consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        self.nnet.load_state_dict(state_dict)
        self.mcts = MCTS(self.g, self.nnet, args)

    def clear(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)

    def make_move(self, color, vertex):
        # no legality check other than the space being empty..
        # no side-effects beyond placing the stone..
        if vertex == PASS:
            return True  # noop
        x, y = vertex
        if self.board[x-1, y-1] == EMPTY:
            self.board[x-1, y-1] = color
            return True
        else:
            return False

    def set_size(self, n):
        self.size = n
        self.clear()

    def set_komi(self, k):
        self.komi = k

    def get_move(self, color):
        canonicalBoard = self.g.getCanonicalForm(self.board, color)
        pi = self.mcts.getActionProb(canonicalBoard, temp=0)
        action = np.random.choice(len(pi), p=pi)
        return (action//self.size + 1, action%self.size + 1)
    
    def analyze(self, color):
        # analyze game, plot strategy and win rate
        # to do: let the engine support nalyze command to show the real-time heat map in GUI like sabaki
        canonicalBoard = self.g.getCanonicalForm(self.board, color)
        pi = self.mcts.getActionProb(canonicalBoard, temp=1.0)
        s = self.g.stringRepresentation(canonicalBoard)
        winrate = []
        for a in range(len(pi)):
            if (s,a) in self.mcts.Qsa:
                winrate.append(self.mcts.Qsa[(s,a)])
            else:
                winrate.append(-1)
        winrate = (np.array(winrate[:-1]).reshape(self.size, self.size)+1)*50
        pi = np.array(pi[:-1]).reshape(self.size, self.size)
        # import matplotlib.pyplot as plt
        # plt.figure(9,figsize=[12.8, 9.6])
        # plt.clf()
        # plt.imshow(pi)
        # plt.colorbar()
        # threshold = np.sort(pi.reshape(-1))[-10]
        # for i in range(g.size):
        #     for j in range(g.size):
        #         if pi[i,j]>threshold:
        #             print(i,j)
        #             plt.text(j-0.3,i-0.05,str(round(winrate[i,j],1)),fontsize=9,color='w')
        # plt.savefig("result_pi.png")
        return pi, winrate

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='path to the neural network checkpoint')
    parser.add_argument('--game_size', type=int, default=15, help='board size')
    parser.add_argument('--numMCTSSims', type=int, default=1000, help='the number of random search for each move')
    a = parser.parse_args()
    args["model_path"] = a.model_path
    args["game_size"] = a.game_size
    args["numMCTSSims"] = a.numMCTSSims

    g = NeuralPlayer(args.game_size)
    e = Engine(g)
    print("GTP engine Ready.")
    while 1:
        x = input().strip()
        r = e.send(x)
        print(r)