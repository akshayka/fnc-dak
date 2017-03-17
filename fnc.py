#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model for the FNC.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import pprint
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import util
from model import Model


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # TODO(akshayka): Add dropout or regularization
    def __init__(self, n_features=1, n_classes=4, method="rnn",
        embed_size=50, hidden_sizes=[50], dropout=0.0, transform_mean=False,
        batch_size=52, unweighted_loss=False, scoring_metrics=None,
        regularizer=None, penalty=0.05, n_epochs=10,
        similarity_metric_feature=None,
        train_embeddings_epoch=10, lr=0.001, output_path=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.method = method
        self.embed_size = embed_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.transform_mean = transform_mean
        self.batch_size = batch_size
        self.unweighted_loss = unweighted_loss
        self.scoring_metrics = scoring_metrics[:-1] \
            if scoring_metrics is not None else None
        self.regularizer=regularizer
        self.penalty=penalty
        self.n_epochs = n_epochs
        self.similarity_metric_feature = similarity_metric_feature
        self.train_embeddings_epoch = train_embeddings_epoch if \
            train_embeddings_epoch is not None else 1 + n_epochs
        self.lr = lr

        self.layers = len(self.hidden_sizes)

        if scoring_metrics is not None:
            try:
               self.degree = int(scoring_metrics[-1])
            except ValueError:
                raise ValueError, "The last argument of -sm must be an integer."

        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = \
                "results/{:%Y%m%d_%H%M%S}_{:}d_{:}L_{:}/".format(
                datetime.now(), self.embed_size, self.layers, self.method)
            os.makedirs(self.output_path)
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.error_output = self.output_path + "errors.txt"
        self.log_output = self.output_path + "log"


class FNCModel(Model):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    """
    SUPPORTED_METHODS = frozenset(["rnn", "gru", "lstm", "bag_of_words",
        "vanilla_bag_of_words", "arora"])
    SUPPORTED_SCORING_METRICS = frozenset(["manhattan", "cosine",
        "soft_cosine"])
    SUPPORTED_SIMILARITY_METRIC_FEATS = frozenset(["cosine", "jaccard"])

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.  Note that when "None" is in a
        placeholder's shape, it's flexible (so we can use different batch sizes
        without rebuilding the model).

        Adds following nodes to the computational graph

        inputs_placeholder: Input placeholder tensor of shape
            (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None)
            type tf.int32

            self.inputs_placeholder
            self.labels_placeholder
        """
        headlines_shape = (None, self.config.embed_size) if \
            self.config.method == "vanilla_bag_of_words" else \
            (None, self.max_headline_len, self.config.n_features)
        bodies_shape = (None, self.config.embed_size) if \
            self.config.method == "vanilla_bag_of_words" else \
            (None, self.max_body_len, self.config.n_features)
        input_dtype = tf.float32 if \
            self.config.method == "vanilla_bag_of_words" else \
            tf.int32

        self.headlines_placeholder = tf.placeholder(input_dtype,
            shape=headlines_shape)
        self.bodies_placeholder = tf.placeholder(input_dtype,
            shape=bodies_shape)
        self.epoch_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.sim_scores_placeholder = tf.placeholder(tf.float32, shape=(None))


    def create_feed_dict(self, headlines_batch, bodies_batch, epoch,
        sim_scores_batch=None, dropout=0.0, labels_batch=None):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            headlines_batch: A batch of headlines.
            bodies_batch: A batch of input bodies.
            epoch: The current epoch.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}
        feed_dict[self.headlines_placeholder] = headlines_batch
        feed_dict[self.bodies_placeholder] = bodies_batch
        if self.config.similarity_metric_feature:
            feed_dict[self.sim_scores_placeholder] = sim_scores_batch
        feed_dict[self.epoch_placeholder] = epoch
        feed_dict[self.dropout_placeholder] = dropout
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict


    def add_embedding(self, input_type, scope):
        """Adds an embedding layer that maps from input tokens (integers) to
        vectors and then concatenates those vectors:

            - Create an embedding tensor and initialize it with
              self.pretrained_embeddings.
            - Use the inputs_placeholder to index into the embeddings tensor,
              resulting in a tensor of shape
              (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings
              tensor to shape (None, max_length, n_features * embed_size).

        Args:
            input_type : str, one of 'headlines' or 'bodies'
        Returns:
            embeddings: tf.Tensor of shape
                        (None, max_length, n_features*embed_size)
        """
        def constant_embeddings():
            return tf.constant(self.pretrained_embeddings, dtype=tf.float32)
        def variable_embeddings(scope):
            with tf.variable_scope(scope):
                return tf.get_variable("embeddings",
                    shape=np.shape(self.pretrained_embeddings),
                    initializer=tf.constant_initializer(
                        self.pretrained_embeddings),
                    dtype=tf.float32)

        # TODO(akshakya): ugly code ...
        if self.config.method == "vanilla_bag_of_words":
            if input_type == "headlines":
                return self.headlines_placeholder
            elif input_type == "bodies":
                return self.bodies_placeholder
            else:
                raise ValueError("Invalid input_type %s" % input_type) 

        embeddings = tf.cond(tf.less(self.epoch_placeholder,
            self.config.train_embeddings_epoch),
            lambda: constant_embeddings(), lambda: variable_embeddings(scope))

        if input_type == "headlines":
            input_embeddings = tf.nn.embedding_lookup(embeddings,
                self.headlines_placeholder)
            max_len = self.max_headline_len
        elif input_type == "bodies":
            input_embeddings = tf.nn.embedding_lookup(embeddings,
                self.bodies_placeholder)
            max_len = self.max_body_len
        else:
            raise ValueError("Invalid input_type %s" % input_type) 

        embeddings_shape = (-1, self.config.embed_size) if \
            self.config.method == "vanilla_bag_of_words" else \
            (-1, max_len, self.config.n_features * self.config.embed_size)
        embeddings = tf.reshape(input_embeddings, embeddings_shape)
        return embeddings


    def add_hidden_op(self, input_type, scope):
        """
        Args:
            input_type : str, one of 'headlines' or 'bodies'
            scope : scope for variables

        Returns:
            hidden: tf.Tensor of shape
                (None, self.config.hidden_sizes[-1]), where None is the current
                batch size
        """
        def sequence_length(x):
            used = tf.sign(tf.reduce_max(tf.abs(x), axis=2))
            seqlen = tf.cast(tf.reduce_sum(used, axis=1), tf.int32)
            return seqlen

        x = self.add_embedding(input_type, scope)
        xav = tf.contrib.layers.xavier_initializer()

        if self.config.method in ["rnn", "gru", "lstm"]:
            seqlen = sequence_length(x)
            cells = []
            inputs = x
            kwargs = {}
            if self.config.method == "rnn":
                cell_type = tf.contrib.rnn.BasicRNNCell
            elif self.config.method == "gru":
                cell_type = tf.contrib.rnn.GRUCell
            elif self.config.method == "lstm":
                cell_type = tf.contrib.rnn.LSTMCell
                kwargs["state_is_tuple"] = True
            for layer, hsz in enumerate(self.config.hidden_sizes):
                cell = cell_type(num_units=hsz)
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                    input_keep_prob=(1-self.dropout_placeholder))
                cells.append(cell)
            if self.config.layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell(cells=cells,
                    state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                output_keep_prob=(1-self.dropout_placeholder))
            # TODO(akshayka): How do we declare an initializer?
            outputs, h = tf.nn.dynamic_rnn(cell=cell, inputs=inputs,
                sequence_length=seqlen, dtype=tf.float32, scope=scope)
            # if cell is a MultiRNNCell, then h is a tuple of hidden states,
            # one per layer; else, it is a single hidden state.
            h = h[-1] if self.config.layers > 1 else h
            # if each cell is an LSTM cell, then h is a tuple of the form
            # (state, hidden_state)
            h = h[1] if self.config.method == "lstm" else h
        elif self.config.method in ["bag_of_words", "arora"]:
            seqlen = sequence_length(x)
            inputs = x
            inputs_shape = inputs.get_shape().as_list()
            # Transformation layers
            if self.config.layers > 1:
                inputs = tf.reshape(inputs, (-1, inputs_shape[2]))
                for layer in range(self.config.layers)[:-1]:
                    local_scope = scope + "/layer" + str(layer)
                    with tf.variable_scope(local_scope):
                        U = tf.get_variable("U",
                            shape=(inputs_shape[2], inputs_shape[2]),
                            initializer=xav)
                        relu_input = tf.matmul(inputs, U)
                        relu_input = tf.nn.dropout(relu_input,
                            keep_prob=(1-self.dropout_placeholder))
                        inputs = tf.nn.relu(relu_input)
                inputs = tf.reshape(inputs,
                    (-1, inputs_shape[1], inputs_shape[2]))
            seqlen_scale = tf.cast(tf.expand_dims(seqlen, axis=1), tf.float32)
            h = tf.divide(tf.reduce_sum(input_tensor=inputs, axis=1),
                seqlen_scale)
            # TODO(akshayka): It would be much more efficient to precompute
            # these sentence embeddings in util. This would require a lot of
            # refactoring, however.
            if self.config.method == "arora":
                pc = self.headlines_pc if input_type == "headlines" \
                    else self.bodies_pc
                h -= tf.multiply(tf.matmul(h, pc), tf.transpose(pc))
            # Average layer
            if self.config.transform_mean:
                with tf.variable_scope(scope + "/transform_mean"):
                    U = tf.get_variable("U", shape=(inputs.get_shape().as_list()[2],
                        self.config.hidden_sizes[-1]), initializer=xav)
                    relu_input = tf.matmul(h, U)
                    relu_input = tf.nn.dropout(relu_input,
                        keep_prob=(1-self.dropout_placeholder))
                    # TODO(akshayka): Experiment with other nonlinearities
                    h = tf.nn.relu(relu_input)
        elif self.config.method == "vanilla_bag_of_words":
            # TODO(akshayka): sorry for this code duplication ...!
            h = x
            if self.config.transform_mean:
                with tf.variable_scope(scope + "/transform_mean"):
                    U = tf.get_variable("U", shape=(h.get_shape().as_list()[-1],
                        self.config.hidden_sizes[-1]), initializer=xav)
                    relu_input = tf.matmul(h, U)
                    relu_input = tf.nn.dropout(relu_input,
                        keep_prob=(1-self.dropout_placeholder))
                    # TODO(akshayka): Experiment with other nonlinearities
                    h = tf.nn.relu(relu_input)
            
        else:
            raise ValueError("Unsuppported method: " + self.config.method)

        return h


    def add_regression_op(self, scores, degree, scope):
        preds = 0
        for d in range(1, degree+1):
            with tf.variable_scope(scope + "/deg_%d" % d):
                m = tf.get_variable("m", shape=[],
                    initializer=tf.constant_initializer(4.0),
                    dtype=tf.float32)
                preds += tf.multiply(m, tf.pow(scores, d))
        with tf.variable_scope(scope + "/deg_%d" % 0):
            b = tf.get_variable("b", shape=[],
                initializer=tf.constant_initializer(0.0),
                dtype=tf.float32)
            preds += b
        return preds


    def add_scoring_metrics_pred_op(self, body_hidden, headline_hidden):
        preds = 0
        if "manhattan" in self.config.scoring_metrics:
            distance = tf.reduce_mean(tf.abs(headline_hidden - body_hidden),
                axis=1)
            # shape (batch_size, 1)
            scores = tf.exp(-1 * distance)
            preds += self.add_regression_op(scores, self.config.degree,
                "prediction_op/manhattan")
        if "cosine" in self.config.scoring_metrics:
            headline_norm = tf.nn.l2_normalize(headline_hidden, dim=1)
            body_norm = tf.nn.l2_normalize(body_hidden, dim=1)
            scores = tf.reduce_sum(tf.multiply(headline_norm, body_norm),
                axis=1)
            preds += self.add_regression_op(scores, self.config.degree,
                "prediction_op/cosine")
        if "soft_cosine" in self.config.scoring_metrics:
            headline_norm = tf.nn.l2_normalize(headline_hidden, dim=1)
            body_norm = tf.nn.l2_normalize(body_hidden, dim=1)
            with tf.variable_scope("prediction_op/soft_cosine"):
                W = tf.get_variable("W", shape=(
                    self.config.hidden_sizes[-1],
                    self.config.hidden_sizes[-1]))
                # scores = (h_2 * W)  h_1^T
                scores = tf.reduce_sum(tf.multiply(
                    tf.matmul(headline_norm, W), body_norm), axis=1)
                preds += self.add_regression_op(scores, self.config.degree,
                    "prediction_op/soft_cosine")
        # shape (batch_size, 1)
        return preds


    def add_prediction_op(self):
        """Adds Ops for prediction (excluding the softmax) to the graph.

        Args:
            headline_transformed: tensor of shape
                (self.config.batch_size, self.config.hidden_sizes[-1])
            headline_transformed: tensor of shape
                (self.config.batch_size, self.config.hidden_sizes[-1])

        Returns:
            preds : tensor of shape
                (self.config.batch_size, self.config.n_classes)
        """
        body_hidden = self.add_hidden_op(input_type='bodies',
            scope=self.body_scope)
        headline_hidden = self.add_hidden_op(input_type='headlines',
            scope=self.headline_scope)

        # TODO(akshayka): Experiment with using the cosine similarity
        # as the similarity metric, instead of the l1 norm?
        if self.config.scoring_metrics is not None:
            preds = self.add_scoring_metrics_pred_op(body_hidden,
                headline_hidden)
        else:
            # TODO(akshayka): append the cosine similarity to pred_input
            # this will need to be a placeholder
            pred_input = tf.concat(axis=1,
                values=[headline_hidden, body_hidden])
            W_hidden_size = 2 * self.config.hidden_sizes[-1]
            if self.config.similarity_metric_feature:
                sim_scores = tf.expand_dims(self.sim_scores_placeholder, axis=1)
                pred_input = tf.concat(axis=1,
                    values=[pred_input, sim_scores])
                W_hidden_size += 1
            with tf.variable_scope("prediction_op"):
                W = tf.get_variable("W", (W_hidden_size,
                    self.config.n_classes))
                b = tf.get_variable("b", (self.config.n_classes),
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)
                preds = tf.matmul(pred_input, W) + b
            return preds


    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        Compute averaged cross entropy loss for the predictions.

        Args:
            preds: A tensor of shape (batch_size, n_classes) containing the
                output of the neural network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """

        labels = self.labels_placeholder
        float_labels = tf.cast(labels, tf.float32)
        if self.config.scoring_metrics is not None:
            loss = tf.abs(float_labels - preds)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=preds,
                    labels=labels)

        weights_per_label = 1.0
        if not self.config.unweighted_loss:
            # related_mask[i] == 1 if labels[i] > 0, 0 otherwise
            related_mask = tf.sign(float_labels)
            # unrelated_mask[i] == 1 if labels[i] == 0, 0 otherwise
            unrelated_mask = tf.abs(related_mask - 1)
            # weights_per_label[i] == 0.25 if labels[i] == 0, 1 otherwise
            weights_per_label = related_mask + 0.25 * unrelated_mask
        loss = tf.multiply(weights_per_label, loss)
        loss = tf.reduce_mean(loss)

        if self.config.regularizer is not None:
            weights = tf.trainable_variables()
            if self.config.regularizer == "l2":
                reg = tf.contrib.layers.l2_regularizer(
                    scale=self.config.penalty)
            elif self.config.regularizer == "l1":
                reg = tf.contrib.layers.l1_regularizer(
                    scale=self.config.penalty)
            else:
                raise ValueError("Invalid regularizer.")
            penalty = tf.contrib.layers.apply_regularization(reg, weights)
            loss += penalty

        return loss


    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op


    def __init__(self, config, max_headline_len, max_body_len,
        pretrained_embeddings, headlines_pc=None, bodies_pc=None,
        verbose=False):
        super(FNCModel, self).__init__(config=config, verbose=verbose)
        self.config = config
        logging.debug("Creating model with method %s", self.config.method)
        self.max_headline_len = max_headline_len
        self.max_body_len = max_body_len
        self.pretrained_embeddings = pretrained_embeddings
        if self.config.method == "arora":
            self.headlines_pc = tf.constant(headlines_pc, dtype=tf.float32)
            self.bodies_pc = tf.constant(bodies_pc, dtype=tf.float32)

        # Defining placeholders.
        self.headlines_placeholder = None
        self.bodies_placeholder = None
        self.labels_placeholder = None

        self.headline_scope = 'headlines'
        self.body_scope = 'bodies' 

        self.build()
