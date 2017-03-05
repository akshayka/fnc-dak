#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN adapted from Assignment 3 Q2
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np
import copy
import pdb
import util
from model import Model
from rnn_cell import RNNCell


logger = logging.getLogger("baseline_model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    def __init__(self, n_features=1, n_classes=4, cell="rnn",
        embed_size=50, hidden_size=50, transform_size=30,
        batch_size=52, n_epochs=10, lr=0.001, output_path=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.cell = cell
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
            self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(
                datetime.now())
            os.makedirs(self.output_path)
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"


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
        labels_placeholder: Labels placeholder tensor of shape (None)
            type tf.int32

            self.inputs_placeholder
            self.labels_placeholder
            self.dropout_placeholder
        """
        self.headlines_placeholder = tf.placeholder(tf.int32,
            shape=(None, self.max_headline_len, self.config.n_features))
        self.bodies_placeholder = tf.placeholder(tf.int32,
            shape=(None, self.max_body_len, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32,
            shape=(None))


    def create_feed_dict(self, headlines_batch, bodies_batch, labels_batch=None):
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
        if headlines_batch is not None:
            feed_dict[self.headlines_placeholder] = headlines_batch
        if bodies_batch is not None:
            feed_dict[self.bodies_placeholder] = bodies_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict


    def add_embedding(self, input_type):
        """Adds an embedding layer that maps from input tokens (integers) to
        vectors and then concatenates those vectors:

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

        Args:
            input_type : str, one of 'headlines' or 'bodies'
        Returns:
            embeddings: tf.Tensor of shape
                        (None, max_length, n_features*embed_size)
        """
        # TODO(akshayka): Train embeddings after N iterations
        embeddings = tf.constant(self.pretrained_embeddings, dtype=tf.float32)
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

        embeddings = tf.reshape(input_embeddings, [-1, max_len,
            self.config.n_features * self.config.embed_size])
        return embeddings


    def add_hidden_op(self, input_type, scope):
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

        Args:
            input_type : str, one of 'headlines' or 'bodies'
            scope : scope for variables

        Returns:
            hidden: tf.Tensor of shape (batch_size, self.config.hidden_size)
        """
        x = self.add_embedding(input_type)
        max_len = self.max_headline_len if input_type == "headlines" \
            else self.max_body_len

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

        h = tf.zeros((tf.shape(x)[0], self.config.hidden_size), tf.float32)
        with tf.variable_scope(self.config.cell + '_' + scope):
            # Upon completion of this loop,
            # h will contain the final hidden representation of the text
            for time_step in range(max_len):
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
        xav = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(scope):
            U = tf.get_variable("U", (self.config.hidden_size,
                self.config.transform_size), initializer=xav)
            transformed_hidden = tf.matmul(hidden, U)
        return transformed_hidden


    def add_prediction_op(self):
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
        headline_hidden = self.add_hidden_op(input_type='headlines',
            scope=self.headline_scope)
        headline_transformed = self.add_transform_op(headline_hidden,
            scope=self.headline_scope)
        
        body_hidden = self.add_hidden_op(input_type='bodies', scope=self.body_scope)
        body_transformed = self.add_transform_op(body_hidden, scope=self.body_scope)
        pred_input = tf.concat(1, [headline_transformed, body_transformed])

        with tf.variable_scope("prediction_op"):
           W = tf.get_variable("W", (2* self.config.transform_size,
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
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)        
        return train_op


    def __init__(self, config, max_headline_len, max_body_len,
        pretrained_embeddings):
        # super(RNNModel, self).__init__(config, report)
        self.config = config
        self.max_headline_len = max_headline_len
        self.max_body_len = max_body_len
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.headlines_placeholder = None
        self.bodies_placeholder = None
        self.labels_placeholder = None

        self.headline_scope = 'headlines'
        self.body_scope = 'bodies' 

        self.build()


def do_train(train_bodies, train_stances, dimension, embedding_path,
    max_headline_len=None, max_body_len=400):
    logging.info("Loading training and dev data ...")
    fnc_data, fnc_data_train, fnc_data_dev = util.load_and_preprocess_fnc_data(
        train_bodies, train_stances)
    if max_headline_len is None:
        max_headline_len = fnc_data_train.max_headline_len
    if max_body_len is None:
        max_body_len = fnc_data_train.max_body_len

    # For convenience, create the word indices map over the entire dataset
    logging.info("Building word-to-index map ...")
    corpus = ([w for bod in fnc_data.bodies for w in bod] +
        [w for headline in fnc_data.headlines for w in headline])
    word_indices = util.process_corpus(corpus)
    logging.info("Building embedding matrix ...")
    embeddings = util.load_embeddings(word_indices, dimension, embedding_path)

    logging.info("Vectorizing data ...")
    # Vectorize and assemble the training data
    headline_vectors = util.vectorize(fnc_data_train.headlines, word_indices,
        max_headline_len)
    body_vectors = util.vectorize(fnc_data.bodies, word_indices, max_body_len)
    training_data = zip(headline_vectors, body_vectors, fnc_data_train.stances)

    # Vectorize and assemble the dev data; note that we use the training
    # maximum length
    headline_vectors = util.vectorize(fnc_data_dev.headlines, word_indices,
        max_headline_len)
    body_vectors = util.vectorize(fnc_data.bodies, word_indices, max_body_len)
    dev_data = zip(headline_vectors, body_vectors, fnc_data_dev.stances)

    config = Config()
    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
       '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(config, max_headline_len, max_body_len, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            logging.info('Fitting ...')
            model.fit(session, saver, training_data, dev_data)
            # Save predictions in a text file.
            logging.info('Outputting ...')
            output = model.output(session, dev_data)
            # TODO(akshayka): Pickle output
            headlines, bodies = output[0]
            indices_to_words = {word_indices[w] : w for w in word_indices}
            headlines = [' '.join(
                util.word_indices_to_words(h, indices_to_words))
                for h in headlines]
            bodies = [' '.join(
                util.word_indices_to_words(b, indices_to_words))
                for b in bodies]
            output = (headlines, bodies, output[1], output[2])

            with open(model.config.eval_output, 'w') as f:
                for headline, body, label, prediction in output:
                    f.write("%s\t%s\tgold:%d\tpred:%d" % (
                        headline, body, label, prediction))


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
        default="glove/glove.6B.50d.txt", help="Path to word vectors file")
    command_parser.add_argument('-d', '--dimension', type=int,
        default=50, help="Dimension of pretrained word vectors")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS.train_bodies, ARGS.train_stances, ARGS.dimension,
            ARGS.embedding_path)
