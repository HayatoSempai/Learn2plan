#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:56:24 2020

@author: yoyoman
"""

class Constraint :
    '''
    Super-class for all types of constraints
    '''
    
    def __init__(self, name, weight):
        '''
        Constructor
        Input :
        name 
        weight : float > 0 corresponds to the weight of the constraint used to compute the final score
        '''
        self.name = name
        self.is_activated = False
        if weight < 0:
            raise Exception('Invalid weight')
        self.weight = weight
    
    def isSatisfied(self):
        pass
    
class checkBlockEmpty(Constraint) :
    '''
    Constraint checking if a bloc has not been played
    '''
    def __init(self) :
        '''
        Constructor
        Attributes :
        i, j the indexes in the board of the block concerned
        '''
        super().__init__()
        self.i
        self.j
    
    def setIndex(self,i,j) :
        '''
        Setter for i, j the indexes in the board of the block concerned
        '''
        self.i = i
        self.j = j
    
    def isActivated(self, Board):
        '''
        Checking if a bloc has been played
        '''
        if Board.state['state'][self.i][self.j] != 1 :
            self.is_activated = True
        else:
            self.is_activated = False
            
class checkHoursPerDay(Constraint) :
    '''
    Constraint checking the number of hours in a day (< value ?)
    '''
    def __init(self) :
        '''
        Constructor
        Attributes :
        max_hours_day the indexes max number of work hours per day
        index of day concerned
        '''
        super().__init__()
        self.max_hours_day
        self.index
    
    def setHours(self, max_hours_day, index) :
        '''
        Setter for max_hours_day and the index of day concerned
        '''
        self.max_hours_day = max_hours_day
        self.index = index
    
    def isActivated(self, Board):
        '''
        Checking if a bloc has been played
        '''
        if sum(Board.state['state'][:, self.index]) < self.max_hours_day :
            self.is_activated = True
        else:
            self.is_activated = False
