import argparse
import logging
import os
import pprint
import sys
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
import numpy as np

import util
from model import Model
from fnc import FNCModel, Config


class AttentionCell(tf.contrib.rnn.GRUCell):
    def __init__(self, num_units, encoder_outputs, scope=None):
        self.encoder_outputs = encoder_outputs
        self.enc_norm = tf.nn.l2_normalize(self.encoder_outputs, dim=2)
        super(AttentionCell, self).__init__(num_units)


    def __call__(self, inputs, state, scope=None):
        gru_out, _ = super(AttentionCell, self).__call__(
            inputs, state, scope)

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Attn"):
                # TODO(akshayka): This initial linear transformation
                # seems strange.
                # ht = _linear(gru_out, self._num_units, True, 1.0)
                W = tf.get_variable("W", shape=(self._num_units,
                    self._num_units))
                ht = tf.matmul(gru_out, W)
                ht = tf.expand_dims(ht, axis=1)
            # TODO(akshayka): We should probably normalize
            # self.encoder_outputs and ht before computing the scores
            ht_norm = tf.nn.l2_normalize(ht, dim=2)
            scores = tf.reduce_sum(self.enc_norm * ht_norm, axis=2,
                keep_dims=True)
            context = tf.reduce_sum(self.encoder_outputs * scores, axis=1)
            with tf.variable_scope("AttnConcat"):
                W = tf.get_variable("W", shape=(2 * self._num_units,
                    self._num_units))
                cat = tf.concat(axis=1, values=[context, gru_out])
                out = tf.nn.relu(tf.matmul(cat, W))
                #out = tf.nn.relu(_linear([context, gru_out], self._num_units,
                #    True, 1.0))
        return (out, out)
        

class Seq2SeqModel(FNCModel):
    def get_rnn_cell(self, cell_type):
        if cell_type == "rnn":
            cell = tf.contrib.rnn.BasicRNNCell
        elif cell_type == "gru":
            cell = tf.contrib.rnn.GRUCell
        elif cell_type == "lstm":
            cell = tf.contrib.rnn.LSTMCell
        elif cell_type == "attention":
            cell = AttentionCell
        else:
            raise ValueError("Unknown cell type %s" % cell_type)
        return cell


    def add_hidden_op(self, scope, old_h=None):
        def sequence_length(x):
            used = tf.sign(tf.reduce_max(tf.abs(x), axis=2))
            seqlen = tf.cast(tf.reduce_sum(used, axis=1), tf.int32)
            return seqlen

        inputs = self.add_embedding(input_type=scope, scope=scope)
        seqlen = sequence_length(inputs)

        if scope == self.body_scope:
            cell = self.get_rnn_cell(self.config.method)(
                num_units=self.config.hidden_sizes[0])
            inputs = tf.reverse_sequence(inputs, seqlen, seq_axis=2)
            outputs, h = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                sequence_length=seqlen, dtype=tf.float32, scope=scope)
            self.encoder_outputs = outputs
        elif scope == self.headline_scope:
            # Uh-ten-SHUN!
            cell = self.get_rnn_cell("attention")(
                num_units=self.config.hidden_sizes[0],
                encoder_outputs=self.encoder_outputs)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, 
                input_keep_prob=(1-self.dropout_placeholder))
            outputs, h = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                sequence_length=seqlen, dtype=tf.float32,
                initial_state=old_h, scope=scope)
        else:
            raise ValueError("Unkown input type %s" % scope)

        return h

    def add_prediction_op(self):
        h_body = self.add_hidden_op(self.body_scope)
        h = self.add_hidden_op(self.headline_scope, h_body)

        if self.config.scoring_metrics is not None:
            preds = self.add_scoring_metrics_pred_op(h_body, h)
        else:
            xav = tf.contrib.layers.xavier_initializer()
            if self.config.similarity_metric_feature:
                sim_scores = tf.expand_dims(self.sim_scores_placeholder, axis=1)
                h = tf.concat(axis=1, values=[h, sim_scores])
                W_hidden_size += 1
            with tf.variable_scope("prediction_op"):
                U = tf.get_variable("U", (self.config.hidden_sizes[0],
                    self.config.n_classes), initializer=xav)
                b = tf.get_variable("b", (self.config.n_classes),
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)
                preds = tf.matmul(h, U) + b
        return preds
