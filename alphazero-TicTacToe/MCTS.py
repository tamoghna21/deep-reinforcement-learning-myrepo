# MCTS algorithm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy
from math import *
import random

def play_a_line(game):
    game1 = copy(game)
    #print("playing a random line..")
    if (game1.score is not None):
        print("game1.state: ", game1.state)
        print("game1.player: ", game1.player)
        print("game1.score: ", game1.score)
        raise ValueError('play a line called with score already 1 or -1 or 0')
    while(game1.score is None):
        possible_actions = game1.available_moves()
        #print("possible_actions: ", possible_actions)
        action = random.choice(possible_actions)
        #print("chosen action: ",action)
        game1.move(action)
        #print("game1.score: ", game1.score)
    return game1.score

class Node:
    def __init__(self, game, mother=None):
        self.game = game
          
        # child nodes
        self.children = {}#This is a dictionary of all child nodes.For each element,key is a move, value is a Node(same class Node)
        # numbers for determining which actions to take next
        self.U = 0

        # visit count
        self.N = 0

        # expected V from MCTS
        self.V = 0

        # keeps track of the guaranteed outcome
        # initialized to None
        # this is for speeding the tree-search up
        # but stopping exploration when the outcome is certain
        # and there is a known perfect play
        self.outcome = self.game.score


        # if game is won/loss/draw
        if self.game.score is not None:
            self.V = self.game.score*self.game.player
            self.U = 0 if self.game.score == 0 else self.V*float('inf')

        # link to previous node
        self.mother = mother
    
    def create_children(self, actions):
        # create a dictionary of children
        games = [ copy(self.game) for a in actions ]

        for action, game in zip(actions, games):
            game.move(action)

        children = { tuple(a):Node(g, self) for a,g in zip(actions, games) }
        #print(children)
        #for a,c in children.items():
            #print("a: ",a)
            #print("c.U: ", c.U)
            #print("c.game.state: ", c.game.state)
        self.children = children
        
    def explore(self):
        if self.game.score is not None:
            raise ValueError("game has ended with score {0:d}".format(self.game.score))
        # start from the top of the node, so current node is self
        current = self
        #print(current.children)#This is a dictionary, key is a move, value is the Node after the resulting move
        #print(current.outcome)
        # explore children of the node
        # to speed things up
        
        # if childen of a node present, then traverse upto the leaf node with the exception of if the leaf node is won or lost
        if current.children and current.outcome is None:
            while current.children and current.outcome is None:
                #print("choosing an action k: ", k)
                children = current.children
                #print("printing all c.U:...")
                #for c in children.values():
                    #print(c.U)
                max_U = max(c.U for c in children.values())# values() method of a dictionary returns only the values(not the keys) of
                                                     # a dictionary, as a list.
                                                     # Therefore, c.U is actually a child node.U
                #print("current max_U ", max_U) 
                actions = [ a for a,c in children.items() if c.U == max_U ]
                
                #print("available actions: ",actions)
                action = random.choice(actions)            
                #print("chosen action: ",action)
            
                # If the final node is won or lost position i.e max_U=inf or -inf, then break; current = children[action] will
                # not be executed, current is the last node before the leaf node
                if max_U == -float("inf"):
                    current.U = float("inf")
                    current.V = 1.0
                    break
            
                elif max_U == float("inf"):
                    current.U = -float("inf")
                    current.V = -1.0
                    break
                
                current = children[action] #This is important,now current node is no more the base node but one of the children nodes
                #print("game state after chosen action: ", current.game.state)
            
                
                
            current.N += 1
        
        # Expand the current node(the final leaf node) if node hasn't been expanded.
        # Then calculate current.V after playing a random line
        if not current.children and current.outcome is None:
            # policy outputs results from the perspective of the next player
            # thus extra - sign is needed
            #print("processing policy")
            
            next_actions = current.game.available_moves()
            #print("next_actions: ",next_actions)
            
            #print("creating children")
            current.create_children(next_actions)
            
            # Play a random line and calculate the leaf node's V; first time it calculates V for the base node, which is not reqd
            current.V = (-1)* current.game.player* play_a_line(current.game)
            #print("current.V: ", current.V)
            
        # now update U and back-prop
        while current.mother:
            #print("mother entered")
            
            mother = current.mother
            mother.N += 1
            # beteen mother and child, the player is switched, extra - sign
            mother.V += (-current.V - mother.V)/mother.N

            #update U for all sibling nodes
            for sibling in mother.children.values():
                #print("sibling.U: ", sibling.U)
                if sibling.U < float("inf") and sibling.U > -float("inf"):#correction from MCTS.py, 'is not' with < , >
                    sibling.U = sibling.V + sqrt(mother.N)/(1+sibling.N)

            current = current.mother


    def next(self, temperature=1.0):
        #print("Finding best move..")
        if self.game.score is not None:
            raise ValueError('game has ended with score {0:d}'.format(self.game.score))

        if not self.children:
            #print(self.game.state)
            raise ValueError('no children found and game hasn\'t ended')
        
        children=self.children
        
        # if there are winning moves, just output those
        max_U = max(c.U for c in children.values())
        
        #print("max_U: ", max_U)

        if max_U == float("inf"):
            prob = [ 1.0 if c.U == float("inf") else 0 for c in children.values()]
            
        else:
            # divide things by maxN for numerical stability
            maxN = max(node.N for node in children.values())+1
            prob = [ (node.N/maxN)**(1/temperature) for node in children.values()]
        
        # normalize the probability
        if sum(prob) > 0:
            prob_norm = [x / sum(prob) for x in prob]
        # if sum is zero, just make things random
        else:
            prob_norm = [1.0/len(children)] * len(children)
        
        nextstate = random.choices(list(children.values()), weights=prob_norm)[0]
        
        return nextstate
