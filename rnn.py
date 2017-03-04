#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN adapted from Assignment 3 Q2
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
import copy
import pdb
import util
from rnn_cell import RNNCell


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self, n_features=1, n_classes=4, dropout=0.5,
        embed_size=300, hidden_size=300, transform_size,
        batch_size=52, n_epochs=10, lr=0.001, output_path=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout = dropout
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.transform_size = transform_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/window/{:%Y%m%d_%H%M%S}/".format(
                datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"


def pad_sequences(data, n_features, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    TODO: In the code below, for every sentence, labels pair in @data,
    (a) create a new sentence which appends zero feature vectors until
    the sentence is of length @max_length. If the sentence is longer
    than @max_length, simply truncate the sentence to be @max_length
    long.
    (b) create a new label sequence similarly.
    token in the original sequence, and a False for every padded input.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * n_features

    for sentence in data:
        # TODO(akshayka): adapt this for our code
        s = copy.deepcopy(sentence)
        # pad_len = max_length - len(s)
        s = (s + [zero_vector for i in range(max_length - len(s))])[0:max_length]
        ret.append(s)
    return ret


class RNNModel(Model):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.  Note that when "None" is in a
        placeholder's shape, it's flexible (so we can use different batch sizes
        without rebuilding the model).

        Adds following nodes to the computational graph

        inputs_placeholder: Input placeholder tensor of shape
            (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape
            (None, self.max_length), type tf.int32
        dropout_placeholder: Dropout value placeholder (scalar),
            type tf.float32

            self.inputs_placeholder
            self.labels_placeholder
            self.dropout_placeholder
        """
        self.inputs_placeholder = tf.placeholder(tf.int32,
            shape=(None, self.max_length, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32,
            shape=(None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
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
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}
        if inputs_batch is not None:
            feed_dict[self.inputs_placeholder] = inputs_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if dropout is not None:
            feed_dict[self.dropout_placeholder] = dropout
        return feed_dict


    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to
        vectors and then concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with
              self.pretrained_embeddings.
            - Use the inputs_placeholder to index into the embeddings tensor,
              resulting in a tensor of shape
              (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings
              tensor to shape (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors.

        Returns:
            embeddings: tf.Tensor of shape
                        (None, max_length, n_features*embed_size)
        """
        # TODO(akshayka): Do not train embeddings, at least not for the first N
        # iterations
        embeddings = tf.Variable(self.pretrained_embeddings)
        input_embeddings = tf.nn.embedding_lookup(embeddings,
            self.inputs_placeholder)
        embeddings = tf.reshape(input_embeddings, [-1, self.max_length,
            self.config.n_features * self.config.embed_size])
        return embeddings

    def add_hidden_op(self, scope):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.pack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability
              (1 - p_drop) as an argument. The keep probability should be set
              to the value of self.dropout_placeholder

        Returns:
            hidden: tf.Tensor of shape (batch_size, self.config.hidden_size)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        if self.config.cell == "rnn":
            cell = RNNCell(self.config.n_features * self.config.embed_size,
                self.config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(self.config.n_features * self.config.embed_size,
                self.config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        xav = tf.contrib.layers.xavier_initializer
        with tf.variable_scope(scope):
            b2 = tf.get_variable("b2", (self.config.n_classes),
                initializer=tf.constant_initializer(0))
            h = tf.zeros((tf.shape(x)[0], self.config.hidden_size), tf.float32)

        with tf.variable_scope('RNN_ ' + scope):
            # Upon completion of this loop,
            # h will contain the final hidden representation of the text
            for time_step in range(self.max_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                x_t = x[:,time_step,:]
                o_t, h = cell(inputs=x_t, state=h, scope=scope)
        return h

    
    def add_transform_op(self, hidden, scope):
        """Adds Ops for the hidden state transformation to the graph.

        Args:
            hidden: A tensor of shape (batch_size, self.config.hidden_size)
                    containing the final hidden state of the RNN.
            scope: The variable scope to use. (For example,
                   'headline' or 'body'.)
        Returns:
            transformed_hidden: A tensor of shape
            (batch_size, self.config.transform_size)
        """
        with tf.variable_scope(scope):
            U = tf.get_variable("U", (self.config.hidden_size,
                self.config.transform_size), initializer=xav())
            transformed_hidden = tf.matmul(hidden, U)
        return transformed_hidden


    def add_prediction_op(self, headline_transformed, body_transformed):
        """Adds Ops for prediction (excluding the softmax) to the graph.

        Args:
            headline_transformed: tensor of shape
                (self.config.batch_size, self.config.transform_size)
            headline_transformed: tensor of shape
                (self.config.batch_size, self.config.transform_size)

        Returns:
            preds : tensor of shape
                (self.config.batch_size, self.config.n_classes)
        """
        pred_input = tf.concat([headline_transformed, body_transformed], axis=0)
        with tf.variable_scope("prediction_op"):
           W = tf.get_variable("W", (self.config.transform_size,
            self.config.n_classes))
           preds = tf.matmul(pred_input, W)
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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(preds,
            self.labels_placeholder)
        loss = tf.reduce_mean(loss)
        return loss


    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        # TODO(akshayka): Experiment with different optimizers
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)        
        return train_op


    def predict_on_batch(self, sess, inputs_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch)
        # TODO(akshayka): populate self.pred / write output function
        # that calls this one
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.max_length = # TODO(akshayka) compute max_length
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.dropout_placeholder = None

        # TODO: pipeline to populate inputs / labels
        self.build()


def do_train(train_bodies, train_stances, dimension, embedding_path):
    # Set up some parameters.
    bodies, stances = util.load_and_preprocess_fnc_data(train_bodies,
        train_stances)
    corpus = ([w for bod in bodies.values() for w in bod] +
        [w for headline in stances[0] for w in headline])
    word_indices = util.process_corpus(corpus)
    embeddings = util.load_embeddings(word_indices, dimension, embedding_path)
    # headline --> rnn --> hidden_output --> transform --> transformed_hidden
    # body --> rnn --> hidden_output --> transform --> transformed_hidden
    # (headline_transformed, body_transformed) --> classifier --> batch pred


def do_evaluate():
    # TODO(akshayka): Implement evaluation function, lean on assignment 3
    pass
    

# TODO(akshayka): Plotting code (loss / gradient size ... ) /
# evaluation of results / etc etc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests RNN model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-tb', '--train-bodies',
        type=argparse.FileType('r'), default="fnc-1-data/train_bodies.csv",
        help="Training data")
    command_parser.add_argument('-ts',
        '--train-stances', type=argparse.FileType('r'),
        default="fnc-1-data/train_stances.csv", help="Training data")
    command_parser.add_argument('-e', '--embedding_path', type=str,
        default="glove/glove.6B.300d.txt", help="Path to word vectors file")
    command_parser.add_argument('-d', '--dimension', type=int,
        default=300, help="Dimension of pretrained word vectors")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(args.train_bodies, args.train_stances, args.dimension,
            args.embedding_path)
