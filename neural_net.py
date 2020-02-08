#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 15:00:28 2020

@author: yoyoman
"""

"""Class to represent the Neural Network."""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os

class NeuralNetwork(object):
    """Represents the Policy and Value Resnet.

    Attributes:
        row: An integer indicating the length of the board row.
        column: An integer indicating the length of the board column.
        action_size: An integer indicating the total number of board squares.
        pi: A TF tensor for the search probabilities.
        v: A TF tensor for the search values.
        states: A TF tensor with the dimensions of the board.
        training: A TF boolean scalar tensor.
        train_pis: A TF tensor for the target search probabilities.
        train_vs: A TF tensor for the target search values.
        loss_pi: A TF tensor for the output of softmax cross entropy on pi.
        loss_v: A TF tensor for the output of mean squared error on v.
        total_loss: A TF tensor to store the addition of pi and v losses.
        train_op: A TF tensor for the train output of the optimizer.
        saver: A TF saver for writing training checkpoints.
        sess: A TF session for running Ops on the Graph.
    """

    def __init__(self, game):
        """Initializes NeuralNetwork with the Resnet network graph."""
        self.row = game.row
        self.column = game.column
        self.action_size = game.action_size
        self.pi = None
        self.v = None
        self.resnet_blocks = 5
        self.learning_rate = 0.01
        self.momentum = 0.9

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states = tf.placeholder(tf.float32,
                                         shape=[None, self.row, self.column])
            self.training = tf.placeholder(tf.bool)

            # Input Layer
            input_layer = tf.reshape(self.states,
                                     [-1, self.row, self.column, 1])

            # Convolutional Block
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                strides=1)

            batch_norm1 = tf.layers.batch_normalization(
                inputs=conv1,
                training=self.training
            )

            relu1 = tf.nn.relu(batch_norm1)

            resnet_in_out = relu1

            # Residual Tower
            for i in range(self.resnet_blocks):
                # Residual Block
                conv2 = tf.layers.conv2d(
                    inputs=resnet_in_out,
                    filters=256,
                    kernel_size=[3, 3],
                    padding="same",
                    strides=1)

                batch_norm2 = tf.layers.batch_normalization(
                    inputs=conv2,
                    training=self.training
                )

                relu2 = tf.nn.relu(batch_norm2)

                conv3 = tf.layers.conv2d(
                    inputs=relu2,
                    filters=256,
                    kernel_size=[3, 3],
                    padding="same",
                    strides=1)

                batch_norm3 = tf.layers.batch_normalization(
                    inputs=conv3,
                    training=self.training
                )

                resnet_skip = tf.add(batch_norm3, resnet_in_out)

                resnet_in_out = tf.nn.relu(resnet_skip)

            # Policy Head
            conv4 = tf.layers.conv2d(
                inputs=resnet_in_out,
                filters=2,
                kernel_size=[1, 1],
                padding="same",
                strides=1)

            batch_norm4 = tf.layers.batch_normalization(
                inputs=conv4,
                training=self.training
            )

            relu4 = tf.nn.relu(batch_norm4)

            relu4_flat = tf.reshape(relu4, [-1, self.row * self.column * 2])

            logits = tf.layers.dense(inputs=relu4_flat, units=self.action_size)

            self.pi = tf.nn.softmax(logits)

            # Value Head
            conv5 = tf.layers.conv2d(
                inputs=resnet_in_out,
                filters=1,
                kernel_size=[1, 1],
                padding="same",
                strides=1)

            batch_norm5 = tf.layers.batch_normalization(
                inputs=conv5,
                training=self.training
            )

            relu5 = tf.nn.relu(batch_norm5)

            relu5_flat = tf.reshape(relu5, [-1, self.action_size])

            dense1 = tf.layers.dense(inputs=relu5_flat,
                                     units=256)

            relu6 = tf.nn.relu(dense1)

            dense2 = tf.layers.dense(inputs=relu6,
                                     units=1)

            self.v = tf.nn.tanh(dense2)

            # Loss Function
            self.train_pis = tf.placeholder(tf.float32,
                                            shape=[None, self.action_size])
            self.train_vs = tf.placeholder(tf.float32, shape=[None])

            self.loss_pi = tf.losses.softmax_cross_entropy(self.train_pis,
                                                           self.pi)
            self.loss_v = tf.losses.mean_squared_error(self.train_vs,
                                                       tf.reshape(self.v,
                                                                  shape=[-1, ]))
            self.total_loss = self.loss_pi + self.loss_v

            # Stochastic gradient descent with momentum
            # global_step = tf.Variable(0, trainable=False)

            # learning_rate = tf.train.exponential_decay(CFG.learning_rate,
            #                                            global_step,
            #                                            20000,
            #                                            0.96,
            #                                            staircase=True)

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                use_nesterov=False)

            self.train_op = optimizer.minimize(self.total_loss)

            # Create a saver for writing training checkpoints.
            self.saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            self.sess = tf.Session()

            # Initialize the session.
            self.sess.run(tf.global_variables_initializer())


class NeuralNetworkWrapper(object):
    """Wrapper class for the NeuralNetwork class.

    Attributes:
        game: An object containing the game state.
        net: An object containing the neural network.
        sess: A TF session for running Ops on the Graph.
    """

    def __init__(self, game):
        """Initializes NeuralNetworkWrapper with game state and TF session."""
        self.game = game
        self.net = NeuralNetwork(self.game)
        self.sess = self.net.sess
        self.epochs = 10
        self.batch_size = 128
        self.record_loss = 1
        self.model_directory = "./models/"
        self.loss_file = "loss.txt"

    def predict(self, state):
        """Predicts move probabilities and state values given a game state.

        Args:
            state: A list containing the game state in matrix form.

        Returns:
            A probability vector and a value scalar
        """
        state = state[np.newaxis, :, :]

        pi, v = self.sess.run([self.net.pi, self.net.v],
                              feed_dict={self.net.states: state,
                                         self.net.training: False})

        return pi[0], v[0][0]

    def train(self, training_data):
        """Trains the network using states, pis and vs from self play games.

        Args:
            training_data: A list containing states, pis and vs
        """
        print("\nTraining the network.\n")

        for epoch in range(self.epochs):
            print("Epoch", epoch + 1)

            examples_num = len(training_data)

            # Divide epoch into batches.
            for i in range(0, examples_num, self.batch_size):
                states, pis, vs = map(list,
                                      zip(*training_data[i:i + self.batch_size]))

                feed_dict = {self.net.states: states,
                             self.net.train_pis: pis,
                             self.net.train_vs: vs,
                             self.net.training: True}

                self.sess.run(self.net.train_op,
                              feed_dict=feed_dict)

                pi_loss, v_loss = self.sess.run(
                    [self.net.loss_pi, self.net.loss_v],
                    feed_dict=feed_dict)

                # Record pi and v loss to a file.
                if self.record_loss:
                    # Create directory if it doesn't exist.
                    if not os.path.exists(self.model_directory):
                        os.mkdir(self.model_directory)

                    file_path = self.model_directory + self.loss_file

                    with open(file_path, 'a') as loss_file:
                        loss_file.write('%f|%f\n' % (pi_loss, v_loss))

        print("\n")

    def save_model(self, filename="current_model"):
        """Saves the network model at the given file path.

        Args:
            filename: A string representing the model name.
        """
        # Create directory if it doesn't exist.
        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)

        file_path = self.model_directory + filename

        print("Saving model:", filename, "at", self.model_directory)
        self.net.saver.save(self.sess, file_path)

    def load_model(self, filename="current_model"):
        """Loads the network model at the given file path.

        Args:
            filename: A string representing the model name.
        """
        file_path = self.model_directory + filename

        print("Loading model:", filename, "from", self.model_directory)
        self.net.saver.restore(self.sess, file_path)
