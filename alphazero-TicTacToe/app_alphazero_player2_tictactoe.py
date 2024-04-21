import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
import random

import MCTSwithRL # MCTS aided by Actor-Critic RL

from Policy_NN import Policy

from copy import copy
import random

# load the weights from file
policy_alphazero = Policy()
policy_alphazero.load_state_dict(torch.load('Policy_alphazero_tictactoe.pth')) 

# Two players are defined below, Policy_Player_MCTS is the MCTS player,
# Random_Player is a random move playing player

def Policy_Player_MCTS(game):
    tree = MCTSwithRL.Node(copy(game))
    for _ in range(50): # explore the tree 50 steps #50
        tree.explore(policy_alphazero) # This will compute all the U s, pick the branch with max U, search, 
                               # expand, backpropagate and increase the visit count
   
    treenext, (v, nn_v, p, nn_p) = tree.next(temperature=0.1) # Asking the tree to choose a next move based on the visit counts
        
    return treenext.game.last_move # returns the move after incrementing the Tree

def Random_Player(game):
    return random.choice(game.available_moves())

from ConnectN import ConnectN

game_setting = {'size':(3,3), 'N':3}


from Play import Play

# as second player
gameplay=Play(ConnectN(**game_setting), 
              player2=Policy_Player_MCTS, 
              player1=None)