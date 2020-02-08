#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:48:56 2020

@author: yoyoman
"""
import numpy as np
from mcts import TreeNode
# Evaluate TODO

class Evaluate(object):
    """Represents the Policy and Value Resnet.

    Attributes:
        current_mcts: An object for the current network's MCTS.
        eval_mcts: An object for the evaluation network's MCTS.
        game: An object containing the game state.
    """

    def __init__(self, current_mcts, eval_mcts, game):
        """Initializes Evaluate with the both network's MCTS and game state."""
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts
        self.game = game
        self.temp_final = 0.001
        self.num_eval_games = 3

    def evaluate(self):
        """Play self-play games between the two networks and record game stats.

        Returns:
            Wins and losses count from the perspective of the current network.
        """
        wins = 0
        losses = 0

        # Self-play loop
        for i in range(self.num_eval_games):
            print("Start Evaluation Self-Play Game:", i, "\n")

            game = self.game.clone()  # Create a fresh clone for each game.
            game_over = False
            value = 0
            node = TreeNode()

#             player = game.current_player

            # Keep playing until the game is in a terminal state.
            while not game_over:
                # MCTS simulations to get the best child node.
                # If player_to_eval is 1 play using the current network
                # Else play using the evaluation network.
#                 if game.current_player == 1:
                best_child = self.current_mcts.search(game, node,
                                                          self.temp_final)
#                 else:
#                     best_child = self.eval_mcts.search(game, node,
#                                                        self.temp_final)

                action = best_child.action
                
                game.play_action(action)  # Play the child node's action.

                game_over, value = game.check_game_over()

                best_child.parent = None
                node = best_child  # Make the child node the root node.
            
            game.print_board()
            final_score = game.evaluate()
            print('Score : ', final_score, ' (% of best score possible : ', np.round(final_score*100/game.maxScore, 2), '%)')

            if value == 1:
                print("win")
                wins += 1
            elif value == -1:
                print("loss")
                losses += 1
            else:
                print("draw")
            print("\n")

        return wins, losses