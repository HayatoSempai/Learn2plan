#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:57:17 2020

@author: yoyoman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:57:48 2020

@author: yoyoman
"""
from board import Board
from constraint import checkBlockEmpty, checkHoursPerDay

from neural_net import NeuralNetworkWrapper
import os
from train import Train


board_test = Board(hours = 20)

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
for index, day in enumerate (days):
    const = checkBlockEmpty('Early evening departure ' + day, 1.2)
    const.setIndex(6, index)
    board_test.addConstraint(const)
    
for index, day in enumerate (days):
    const = checkBlockEmpty('Late morning start ' + day, 1.5)
    const.setIndex(0, index)
    board_test.addConstraint(const)

for index, day in enumerate (days):
    const = checkHoursPerDay('Nb hours per day ' + day, 2)
    const.setHours(4, index)
    board_test.addConstraint(const)
    
for index, day in enumerate (days):
    const = checkBlockEmpty('Lunch break ' + day, 3.5)
    const.setIndex(4, index)
    board_test.addConstraint(const)
    
board_test.maxScore = board_test.random_score()
    
def learn2plan(board):  
    # Initialize the game object with the chosen game.
    load_model = 1
    net = NeuralNetworkWrapper(board)
    model_directory = "./models/"

    # Initialize the network with the best model.
    if load_model:
        file_path = model_directory + "best_model.meta"
        if os.path.exists(file_path):
            net.load_model("best_model")
        else:
            print("Trained model doesn't exist. Starting from scratch.")
    else:
        print("Trained model not loaded. Starting from scratch.")
    
    train = Train(board, net)
    train.start()
    scores = train.scores
    print('Done')
    return scores

a = learn2plan(board_test)