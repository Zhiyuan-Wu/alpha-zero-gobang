import logging
import math

import numpy as np
import time

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.gpu_accumulate_time = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s] * self.args.endGameRewardWeight

        if s not in self.Ps:
            # leaf node
            _record_time = time.time()
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.gpu_accumulate_time += time.time() - _record_time
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

class batch_MCTS():
    """
    This class handles the MCTS tree. allow batch evluation
    """

    def __init__(self, game, args, shared_Ps, shared_Es, shared_Vs, query_buffer, identifier):
        self.game = game
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = shared_Ps  # stores initial policy (returned by neural net), shared between workers.

        self.Es = shared_Es  # stores game.getGameEnded ended for board s, shared between workers.
        self.Vs = shared_Vs  # stores game.getValidMoves for board s, shared between workers.
        self.qb = query_buffer # stores board to be evaluated
        self.identifier = identifier # worker id

        self.board = game.getInitBoard()
        self.player = 1
        self.episodeStep = 0
        self.game_record = []
        self.search_path = []
        self.current_state = game.getCanonicalForm(self.board, self.player)
        self.current_value = None

        self.search_count = 0
        self.total_search_depth = 0

    def reset(self):
        ''' Reset the tree state, ready for new search (under same policy). Note that this will not reset the shared content (Ps, Es, Vs).
        '''
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()

        self.board = self.game.getInitBoard()
        self.player = 1
        self.episodeStep = 0
        self.game_record = []
        self.search_path = []
        self.current_state = self.game.getCanonicalForm(self.board, self.player)
        self.current_value = None

    def extend(self):
        ''' excute a forward search, stop at a leaf node (non-evaluated node or terminal-node).
        '''
        self.current_state = self.game.getCanonicalForm(self.board, self.player)
        while 1:
            s = self.game.stringRepresentation(self.current_state)

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(self.current_state, 1)
            if self.Es[s] != 0:
                # terminal node
                self.current_value = -self.Es[s] * self.args.endGameRewardWeight
                break
            
            if s not in self.Ps:
                self.qb.append([self.identifier, self.current_state, s])
                self.current_value = None
                self.Ns[s] = 0
                break

            if s not in self.Ns:
                self.Ns[s] = 0
            
            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                    else:
                        u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            next_s, next_player = self.game.getNextState(self.current_state, 1, a)
            self.current_state = self.game.getCanonicalForm(next_s, next_player)

            self.search_path.append((s, a))

    def backprop(self):
        ''' after self.current_value be set (either by extend() for terminal-node, or outside controller for non-evaluated node), update tree statics
        '''
        self.search_count += 1
        self.total_search_depth += len(self.search_path)
        for s,a in self.search_path:
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + self.current_value) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1

            else:
                self.Qsa[(s, a)] = self.current_value
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1
            self.current_value = -self.current_value
        self.search_path.clear()

    