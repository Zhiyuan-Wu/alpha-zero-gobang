from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
from cfunc.cfunc import cgetGameEnded

class GobangGame():
    def __init__(self, n=15, nir=5):
        self.n = n
        self.n_in_row = nir

    def getInitBoard(self):
        # return initial board (numpy board)
        b = np.zeros((self.n, self.n), dtype=np.int8)
        # b = np.zeros((self.n, self.n), dtype=np.int32)
        return b

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        board = board.copy()
        if action == self.n * self.n:
            return (board, -player)
        x, y = (int(action / self.n), action % self.n)
        board[x, y] = player
        return (board, -player)

    def getValidMoves(self, board, player):
        # do not allow pass
        valids = np.zeros((self.getActionSize(),), dtype=np.int8)
        valids[:-1] = board.reshape([-1]) == 0
        return valids

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        # n = self.n_in_row
        # for w in range(self.n):
        #     for h in range(self.n):
        #         if (w in range(self.n - n + 1) and board[w][h] != 0 and
        #                 len(set(board[i][h] for i in range(w, w + n))) == 1):
        #             return board[w][h]
        #         if (h in range(self.n - n + 1) and board[w][h] != 0 and
        #                 len(set(board[w][j] for j in range(h, h + n))) == 1):
        #             return board[w][h]
        #         if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and board[w][h] != 0 and
        #                 len(set(board[w + k][h + k] for k in range(n))) == 1):
        #             return board[w][h]
        #         if (w in range(self.n - n + 1) and h in range(n - 1, self.n) and board[w][h] != 0 and
        #                 len(set(board[w + l][h - l] for l in range(n))) == 1):
        #             return board[w][h]
        # if np.any(board==0):
        #     return 0
        # return 1e-4

        return cgetGameEnded(board.astype(np.int32))

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player==1:
            return board
        else:
            return -board
        # return player * board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tobytes()

    @staticmethod
    def display(board):
        n = board.shape[0]

        print("    ", end="")
        for y in range(n):
            print(str(y)+" ", end="")
        print("")
        print("  "+"-"*(2*n+3))
        for y in range(n):
            print(y, "| ", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                if piece == -1:
                    print("b ", end="")
                elif piece == 1:
                    print("W ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")
        print("  "+"-"*(2*n+3))
