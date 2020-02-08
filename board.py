#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:36:27 2020

"""
import math
import numpy as np
from copy import deepcopy
import os

class Board :
    def __init__(self, row = 7, column = 5, other_players = [], hours = 20):
        '''
        Constructor => Initialize the timetable
        Input :
        row : number of row in the desired timetable - Default = 7
        column : number of days in the desired timetable - Default = 5
        other_players : information about the already occupied time blocks (if more than 1 person) - Default = []
        hours : number of work hours left to place - Default = 20
        
        Attributes :
        row : number of row in the board
        column : number of colums in the board
        state : dict - 
                'state' : 2D matrix containing the information about the current game 
                'other_players' 2D matrix containing the information about the blocks already occupied by other players
        Constraints : list of constraints
        hours : number of blocks left to place
        maxScore : max score possible knowing the constraints
        '''
        
        self.row = row
        self.column = column
        self.action_size = self.row * self.column
        
        if len(other_players) == 0:
            other_players = np.zeros([row,column])
        
        if other_players.shape != (row,column):
            raise Exception('Invalid input')
        
        self.state = {'state' : np.zeros([row,column]), 'other_players' : other_players}
        
        self.Constraints = []
        self.hours = hours
        self.validMoves =  np.c_[np.ones(row*column),np.array([[a, b] for a in np.arange(row) for b in np.arange(column) ])]
        self.maxScore = 100
    
    def clone(self):
        """Creates a deep clone of the board object.

        Returns:
            the cloned game object, with its conserved attributes
        """
        board_clone = Board(hours = self.hours)
        board_clone.state = deepcopy(self.state)
        board_clone.Constraints = self.Constraints
        board_clone.row = self.row
        board_clone.column = self.column
        board_clone.maxScore = self.maxScore
        return board_clone
    
    def play_action(self, index):
        '''
        Adds a work hour for the current player
        Input :
        index, a tuple [a, i, j] with a = value representing wether a move is valid (1) or not (0)
                                        i, j : indexes in the board of the block to place
        '''
        i, j = int(index[1]), int(index[2])
        # Checking if all the moves have been placed
#         if self.hours == 0:
#             raise Exception('All moves already placed')
        # Checking if the move is valid
#         elif index[0] == 0:
#             print(index)
#             raise Exception('Invalid move')
        # Playing => Adding a 1 to the board + Adding a 1 to the previous moves board
#         else:
        self.state['state'][i][j] = 1
        self.state['other_players'][i][j] += 1
        self.hours -= 1
    
    def get_valid_moves(self):
        '''
        Returns the valid moves = the indexes of null values in the board
        
        Returns :
        A list of tuples [a, i, j] with a = value representing wether a move is valid (1) or not (0)
                                        i, j : indexes in the board
        '''
        
        index_valid_moves = np.where(self.state['state'] == 0)
        tuples = np.stack((index_valid_moves[0], index_valid_moves[1]), axis = -1)
        
        is_valid = np.ones(len(tuples))
        A = np.c_[is_valid,tuples]
        
        for k in self.validMoves :
            if (k == A).all(1).any():
                pass
            else :
                k[0] = 0
                
        return self.validMoves
    
    def check_game_over(self):
        '''
        Checks if the agent "won" or "lost" the game = if enough contraints were respected
        In this case : consider that 40% of contraints respected is a win, else is a loss
        
        Returns:
            A bool representing the game over state (when all the hours have been placed)
            An integer action value. (win: 1, loss: -1, draw: 0)
        '''
        eval = self.evaluate()
        
        if self.hours == 0:
            if eval < self.maxScore:
                return True, 1
                 
            else:
                return True, -1
        else:
            # Returning current score (value between -1 and 1 : the closer to 1, the better the score - 1 = best score possible)
            if eval < self.maxScore:
                return False, 1
            else :
                
                return False, -1 + 2*(np.random.random() + eval)/(self.maxScore + 1)
        
        
    
    def print_board(self):
        '''
        Prints the board
        '''
        print(self.state['state'])
        return
    
    def getBoard(self):
        '''
        Getter for self.state
        '''
        return self.state
    
    def getHours(self):
        '''
        Getter for self.hours
        '''
        return self.hours
    
    def addConstraint(self, constraint) :
        '''
        Add a constraint to the constraint list
        '''
        # Adding constraint
        self.Constraints.append(constraint)
        # Adding weight to max score 
        #self.maxScore += constraint.weight
        
    def randomPlay(self):
        '''
        Initializes the board to a random state (playing all moves)
        
        TO BE REVEIWED : DOES NOT ONLY PLAY VALID MOVES
        '''
        validMoves = self.get_valid_moves()
        index = np.arange(np.shape(validMoves)[0])
        pair = np.random.choice(index)
        self.play_action(np.array([1, int(validMoves[pair][1]),int(validMoves[pair][2])]))
        
    def clearConstraints(self):
        '''
        Remove all constraints from the constraints list
        '''
        self.Constraints = []
        
    def evaluate(self):
        '''
        Computes the score of the current player
        '''
        score = 0
        # Weighted sum of all activated constraints
        for const in self.Constraints :
            const.isActivated(self)
            if const.is_activated == True:
                score += const.weight
        return score
    
    def reset(self ,row = 7, column = 5, other_players = [], hours = 20):
        
        self.row = row
        self.column = column
        self.action_size = self.row * self.column
        
        if len(other_players) == 0:
            other_players = np.zeros([row,column])
        
        if other_players.shape != (row,column):
            raise Exception('Invalid input')
        
        self.state = {'state' : np.zeros([row,column]), 'other_players' : other_players}
        
        self.hours = hours
        self.validMoves =  np.c_[np.ones(row*column),np.array([[a, b] for a in np.arange(row) for b in np.arange(column) ])]
        
    def random_score(self):
        while self.hours > 0 :
            self.randomPlay()
        score = self.evaluate()
        
        self.reset()
        
        return score