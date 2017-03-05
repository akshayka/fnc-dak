#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic RNN cell, adapted from PS3 Q2(c)
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np


class RNNCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope):
        """Updates the state using the previous @state and @inputs.
        Remember the RNN equations are:

        h_t = sigmoid(x_t W_x + h_{t-1} W_h + b)

        TODO: In the code below, implement an RNN cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_x, W_h, b to be variables of the apporiate shape
              using the `tf.get_variable' functions. Make sure you use
              the names "W_x", "W_h" and "b"!
            - Compute @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        with tf.variable_scope(scope):
            H = self.state_size
            D = self.input_size
            W_h = tf.get_variable("W_h", [H, H],
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            W_x = tf.get_variable("W_x", [D, H],
                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            b = tf.get_variable("b", [H],
                initializer=tf.constant_initializer(0.0), dtype=tf.float64)
            theta = tf.matmul(inputs, W_x) + tf.matmul(state, W_h) + b
            # Upon hitting padding, propagate the old state forward
            # indicator_i = 1[x[i] != 0], x[i] the ith row of inputs
            indicator = tf.minimum(
                tf.ceil(tf.reduce_sum(tf.abs(inputs), axis=1)), 1)
            new_state = (tf.nn.sigmoid(theta) * indicator +
                state * (1 - indicator))
        output = new_state
        return output, new_state
