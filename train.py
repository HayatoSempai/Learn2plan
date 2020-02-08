#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:42:07 2020

@author: yoyoman
"""

"""Class to train the Neural Network."""

import numpy as np

from mcts import MonteCarloTreeSearch, TreeNode
from neural_net import NeuralNetworkWrapper
from evaluate import Evaluate
from copy import deepcopy

class Train(object):
    """Class with functions to train the Neural Network using MCTS.

    Attributes:
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, game, net):
        """Initializes Train with the board state and neural network."""
        self.game = game
        self.net = net
        self.eval_net = NeuralNetworkWrapper(game)
        self.num_iterations = 10
        self.num_games = 5
        self.eval_win_rate = 0.55
        self.temp_thresh = 10
        self.temp_init = 1
        self.temp_final = 0.001
        self.config_game = 2
        self.scores = []
                
    def start(self):
        """Main training loop."""
        for i in range(self.num_iterations):
            print("Iteration", i + 1)

            training_data = []  # list to store self play states, pis and vs

            for j in range(self.num_games):
                print("Start Training Self-Play Game", j + 1)
                game = self.game.clone()  # Create a fresh clone for each game.
                self.play_game(game, training_data)
                print(game.evaluate())
                self.scores.append(game.evaluate())

            # Save the current neural network model.
            self.net.save_model()

            # Load the recently saved model into the evaluator network.
            self.eval_net.load_model()

            # Train the network using self play values.
            self.net.train(training_data)

            # Initialize MonteCarloTreeSearch objects for both networks.
            current_mcts = MonteCarloTreeSearch(self.net)
            eval_mcts = MonteCarloTreeSearch(self.eval_net)

            ''' TO BE COMPLETED ! '''
            evaluator = Evaluate(current_mcts=current_mcts, eval_mcts=eval_mcts, game=self.game)
            wins, losses = evaluator.evaluate()
#             wins, losses = 1, 1

            print("wins:", wins)
            print("losses:", losses)

            num_games = wins + losses

            if num_games == 0:
                win_rate = 0
            else:
                win_rate = wins / num_games

            print("win rate:", win_rate)

            if win_rate > self.eval_win_rate:
                # Save current model as the best model.
                print("New model saved as best model.")
                self.net.save_model("best_model")
            else:
                print("New model discarded and previous model loaded.")
                # Discard current model and use previous best model.
                self.net.load_model()

    def play_game(self, game, training_data):
        """Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game: An object containing the game state.
            training_data: A list to store self play states, pis and vs.
        """
        mcts = MonteCarloTreeSearch(self.net)

        game_over = False
        value = 0
        self_play_data = []
        count = 0

        node = TreeNode()

        # Keep playing until the game is in a terminal state.
        while not game_over:
            # MCTS simulations to get the best child node.
            if count < self.temp_thresh:
                best_child = mcts.search(game, node, self.temp_init)
            else:
                best_child = mcts.search(game, node, self.temp_final)

            # Store state, prob and v for training.
            self_play_data.append([deepcopy(game.state['state']),
                                   deepcopy(best_child.parent.child_psas),
                                   0])

            action = best_child.action
            game.play_action(action)  # Play the child node's action.
            count += 1

            
            ''' TO BE COMPLETED !! '''
            game_over, value = game.check_game_over()

            best_child.parent = None
            node = best_child  # Make the child node the root node.

        # Update v as the value of the game result.
        for game_state in self_play_data:
            value = -value
            game_state[2] = value
            self.augment_data(game_state, training_data, game.row, game.column)

    def augment_data(self, game_state, training_data, row, column):
        """Loop for each self-play game.

        Runs MCTS for each game state and plays a move based on the MCTS output.
        Stops when the game is over and prints out a winner.

        Args:
            game_state: An object containing the state, pis and value.
            training_data: A list to store self play states, pis and vs.
            row: An integer indicating the length of the board row.
            column: An integer indicating the length of the board column.
        """
        state = deepcopy(game_state[0])
        psa_vector = deepcopy(game_state[1])

        if self.config_game == 2 or self.config_game == 1:
            training_data.append([state, psa_vector, game_state[2]])
        else:
            psa_vector = np.reshape(psa_vector, (row, column))

            # Augment data by rotating and flipping the game state.
            for i in range(4):
                training_data.append([np.rot90(state, i),
                                      np.rot90(psa_vector, i).flatten(),
                                      game_state[2]])

                training_data.append([np.fliplr(np.rot90(state, i)),
                                      np.fliplr(
                                          np.rot90(psa_vector, i)).flatten(),
                                      game_state[2]])