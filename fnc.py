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
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import util
from model import Model


logger = logging.getLogger("baseline_model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    # TODO(akshayka): Add dropout or regularization
    def __init__(self, n_features=1, n_classes=4, method="rnn",
        train_inputs=False, embed_size=50, hidden_size=50,
        batch_size=52, n_epochs=10, lr=0.001, output_path=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.method = method
        self.train_inputs = train_inputs
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            ti_str = "_ti" if self.train_inputs else ""
            self.output_path = \
                "results/{:%Y%m%d_%H%M%S}_{:}d_{:}d_{:}{:}/".format(
                datetime.now(), embed_size, hidden_size, method, ti_str)
            os.makedirs(self.output_path)
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"


class FNCModel(Model):
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
        if not self.config.train_inputs:
            embeddings = tf.constant(self.pretrained_embeddings,
                dtype=tf.float32)
        else:
            with tf.variable_scope(scope):
                embeddings = tf.get_variable("embeddings",
                    shape=np.shape(self.pretrained_embeddings),
                    initializer=tf.constant_initializer(
                        self.pretrained_embeddings), dtype=tf.float32)

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
        """
        Args:
            input_type : str, one of 'headlines' or 'bodies'
            scope : scope for variables

        Returns:
            hidden: tf.Tensor of shape (batch_size, self.config.hidden_size)
        """
        x = self.add_embedding(input_type, scope)
        max_len = self.max_headline_len if input_type == "headlines" \
            else self.max_body_len

        used = tf.sign(tf.reduce_max(tf.abs(x), axis=2))
        seqlen = tf.cast(tf.reduce_sum(used, axis=1), tf.int32)
        xav = tf.contrib.layers.xavier_initializer()
        if self.config.method in ["rnn", "gru"]:
            if self.config.method == "rnn":
                cell = tf.nn.rnn_cell.BasicRNNCell(
                    num_units=self.config.hidden_size)
            elif self.config.method == "gru":
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.hidden_size)
            _, h = tf.nn.dynamic_rnn(cell=cell, inputs=x,
                sequence_length=seqlen, dtype=tf.float32, scope=scope)
        elif self.config.method == "bag_of_words":
            # (batch_size, seq_length, embed_dim) -> (batch_size, embed_dim)
            seqlen_scale = tf.cast(tf.expand_dims(seqlen, axis=1), tf.float32)
            mean = tf.divide(tf.reduce_sum(input_tensor=x, axis=1),
                seqlen_scale)
            with tf.variable_scope(scope):
                U = tf.get_variable("U", (self.config.embed_size,
                    self.config.hidden_size), initializer=xav)
                h = tf.matmul(mean, U)
        else:
            raise ValueError("Unsuppported method: " + self.config.method)

        return h


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
        body_hidden = self.add_hidden_op(input_type='bodies',
            scope=self.body_scope)
        pred_input = tf.concat(1, [headline_hidden, body_hidden])

        with tf.variable_scope("prediction_op"):
           W = tf.get_variable("W", (2 * self.config.hidden_size,
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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds,
            labels=self.labels_placeholder)
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
        pretrained_embeddings, verbose):
        super(FNCModel, self).__init__(verbose=verbose)
        self.config = config
        logging.debug("Creating model with method %s", self.config.method)
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


def do_train(train_bodies, train_stances, dimension, embedding_path, config,
    max_headline_len=None, max_body_len=None, verbose=False):
    logging.info("Loading training and dev data ...")
    fnc_data, fnc_data_train, fnc_data_dev = util.load_and_preprocess_fnc_data(
        train_bodies, train_stances)
    logging.info("%d training examples", len(fnc_data_train.headlines))
    logging.info("%d dev examples", len(fnc_data_dev.headlines))
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
    body_vectors = util.vectorize(fnc_data_train.bodies, word_indices,
        max_body_len)
    training_data = zip(headline_vectors, body_vectors, fnc_data_train.stances)

    # Vectorize and assemble the dev data; note that we use the training
    # maximum length
    headline_vectors = util.vectorize(fnc_data_dev.headlines, word_indices,
        max_headline_len)
    body_vectors = util.vectorize(fnc_data_dev.bodies, word_indices,
        max_body_len)
    dev_data = zip(headline_vectors, body_vectors, fnc_data_dev.stances)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
       '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = FNCModel(config, max_headline_len, max_body_len, embeddings,
            verbose)
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
            output = zip(headlines, bodies, output[1], output[2])

            with open(model.config.eval_output, 'w') as f:
                for headline, body, label, prediction in output:
                    f.write("%s\t%s\tgold:%d\tpred:%d" % (
                        headline, body, label, prediction))


# TODO(akshayka): Plotting code (loss / gradient size ... ) /
# evaluation of results / etc etc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains and tests FNCModel")
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser("train", help="")
    command_parser.add_argument("-tb", "--train-bodies",
        type=argparse.FileType("r"), default="fnc-1-data/train_bodies.csv",
        help="Training data")
    command_parser.add_argument("-ts",
        "--train-stances", type=argparse.FileType("r"),
        default="fnc-1-data/train_stances.csv", help="Training data")
    command_parser.add_argument("-d", "--dimension", type=int,
        default=50, help="Dimension of pretrained word vectors")
    command_parser.add_argument("-mhl", "--max_headline_len", type=int,
        default=None, help="maximum number of words per headline; if None, "
        "inferred from training data")
    command_parser.add_argument("-mbl", "--max_body_len", type=int,
        default=None, help="maximum number of words per body; if None, "
        "inferred from training data")
    command_parser.add_argument("-hd", "--hidden_size", type=int,
        default=50, help="Dimension of hidden represntation")
    command_parser.add_argument("-m", "--method", type=str,
        default="bag_of_words", help="Input embedding method")
    command_parser.add_argument("-ti", "--train_inputs", action="store_true",
        default=False)
    command_parser.add_argument("-b", "--batch_size", type=int,
        default=52)
    command_parser.add_argument("-v", "--verbose", action="store_true",
        default=False)
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    embedding_path = "glove/glove.6B.%dd.txt" % ARGS.dimension
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        config = Config(method=ARGS.method, train_inputs=ARGS.train_inputs,
            embed_size=ARGS.dimension, hidden_size=ARGS.hidden_size,
            batch_size=ARGS.batch_size)
        ARGS.func(train_bodies=ARGS.train_bodies,
            train_stances=ARGS.train_stances,
            dimension=ARGS.dimension,
            embedding_path=embedding_path, config=config,
            max_headline_len=ARGS.max_headline_len,
            max_body_len=ARGS.max_body_len,
            verbose=ARGS.verbose)
