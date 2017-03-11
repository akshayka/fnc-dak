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
        embed_size=50, hidden_sizes=[50], dropout=0.0, transform_mean=False,
        batch_size=52, unweighted_loss=False, similarity_metric=False,
        regularizer=None, penalty=0.05, n_epochs=10,
        train_embeddings_epoch=10, lr=0.001, output_path=None):
        self.n_features = n_features
        self.n_classes = n_classes
        self.method = method
        self.train_embeddings_epoch = train_embeddings_epoch
        self.embed_size = embed_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.transform_mean = transform_mean
        self.batch_size = batch_size
        self.unweighted_loss = unweighted_loss
        self.similarity_metric = similarity_metric
        self.regularizer=regularizer
        self.penalty=penalty
        self.n_epochs = n_epochs
        self.lr = lr

        self.layers = len(self.hidden_sizes)

        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = \
                "results/{:%Y%m%d_%H%M%S}_{:}d_{:}d_{:}_te_{:}/".format(
                datetime.now(), self.embed_size, self.layers, self.method,
                train_embeddings_epoch)
            os.makedirs(self.output_path)
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"


class FNCModel(Model):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    """
    SUPPORTED_METHODS = frozenset(["rnn", "gru", "lstm", "bag_of_words"])

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
        self.headlines_placeholder = tf.placeholder(tf.int32,
            shape=(None, self.max_headline_len, self.config.n_features))
        self.bodies_placeholder = tf.placeholder(tf.int32,
            shape=(None, self.max_body_len, self.config.n_features))
        self.epoch_placeholder = tf.placeholder(tf.int32)
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.int32,
            shape=(None))


    def create_feed_dict(self, headlines_batch, bodies_batch, epoch,
        dropout=0.0, labels_batch=None):
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

        embeddings = tf.reshape(input_embeddings, [-1, max_len,
            self.config.n_features * self.config.embed_size])
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
        seqlen = sequence_length(x)
        xav = tf.contrib.layers.xavier_initializer()
        if self.config.method in ["rnn", "gru", "lstm"]:
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
            if layers > 1:
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
        elif self.config.method == "bag_of_words":
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
            # Average layer
            if self.config.transform_mean:
                with tf.variable_scope(scope + "/transform_mean"):
                    U = tf.get_variable("U", shape=(inputs.get_shape().as_list()[2],
                        self.config.hidden_sizes[-1]), initializer=xav)
                    relu_input = tf.matmul(h, U)
                    relu_input = tf.nn.dropout(relu_input,
                        keep_prob=(1-self.dropout_placeholder))
                    h = tf.nn.relu(relu_input)
        else:
            raise ValueError("Unsuppported method: " + self.config.method)

        return h


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
        headline_hidden = self.add_hidden_op(input_type='headlines',
            scope=self.headline_scope)
        body_hidden = self.add_hidden_op(input_type='bodies',
            scope=self.body_scope)

        # TODO(akshayka): Experiment with using the cosine similarity
        # as the similarity metric, instead of the l1 norm?
        if self.config.similarity_metric:
            # TODO(akshayka): reduce_mean vs reduce_sum?
            distance = tf.reduce_mean(tf.abs(headline_hidden - body_hidden),
                axis=1)
            # shape (batch_size, 1)
            scores = tf.exp(-1 * distance)
            with tf.variable_scope("prediction_op"):
                m = tf.get_variable("m", shape=[], 
                    initializer=tf.constant_initializer(4.0),
                    dtype=tf.float32)
                b = tf.get_variable("b", shape=[],
                    initializer=tf.constant_initializer(0.0), dtype=tf.float32)
                preds = tf.multiply(m, scores) + b
            # shape (batch_size, 1)
            # TODO(akshayka): Should these be rounded to the true prediction or not?
            return preds
        else:
            # TODO(akshayka): append the cosine similarity to pred_input
            # this will need to be a placeholder
            pred_input = tf.concat(axis=1,
                values=[headline_hidden, body_hidden])
            with tf.variable_scope("prediction_op"):
               W = tf.get_variable("W", (2 * self.config.hidden_sizes[-1],
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
        if self.config.similarity_metric:
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
        pretrained_embeddings, verbose):
        super(FNCModel, self).__init__(config=config, verbose=verbose)
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
    max_headline_len=None, max_body_len=None, verbose=False, 
    include_stopwords=True, weight_embeddings=False):
    logging.info("Loading training and dev data ...")
    fnc_data, fnc_data_train, fnc_data_dev = util.load_and_preprocess_fnc_data(
        train_bodies, train_stances, include_stopwords)
    logging.info("%d training examples", len(fnc_data_train.headlines))
    logging.info("%d dev examples", len(fnc_data_dev.headlines))
    if max_headline_len is None:
        max_headline_len = fnc_data_train.max_headline_len
    if max_body_len is None:
        max_body_len = fnc_data_train.max_body_len
    logging.info("Max headline length: %d", max_headline_len)
    logging.info("Max body length: %d", max_body_len)

    # For convenience, create the word indices map over the entire dataset
    logging.info("Building word-to-index map ...")
    corpus = ([w for bod in fnc_data.bodies for w in bod] +
        [w for headline in fnc_data.headlines for w in headline])
    word_indices = util.process_corpus(corpus)
    logging.info("Building embedding matrix ...")
    embeddings, known_words = util.load_embeddings(word_indices=word_indices,
        dimension=dimension, embedding_path=embedding_path,
        weight_embeddings=weight_embeddings)

    logging.info("Vectorizing data ...")
    # Vectorize and assemble the training data
    headline_vectors = util.vectorize(fnc_data_train.headlines, word_indices,
        known_words, max_headline_len)
    body_vectors = util.vectorize(fnc_data_train.bodies, word_indices,
        known_words, max_body_len)
    training_data = zip(headline_vectors, body_vectors, fnc_data_train.stances)

    # Vectorize and assemble the dev data; note that we use the training
    # maximum length
    dev_headline_vectors = util.vectorize(fnc_data_dev.headlines, word_indices,
        known_words, max_headline_len)
    dev_body_vectors = util.vectorize(fnc_data_dev.bodies, word_indices,
        known_words, max_body_len)
    dev_data = zip(dev_headline_vectors, dev_body_vectors, fnc_data_dev.stances)

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
            logging.info('Outputting ...')
            output = model.output(session, dev_data)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains and tests FNCModel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser("train", help="Run do_train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ------------------------ Input Data ------------------------
    command_parser.add_argument("-tb", "--train_bodies",
        type=argparse.FileType("r"), default="fnc-1-data/train_bodies.csv",
        help="Training data")
    command_parser.add_argument("-ts",
        "--train_stances", type=argparse.FileType("r"),
        default="fnc-1-data/train_stances.csv", help="Training data")
    command_parser.add_argument("-d", "--dimension", type=int,
        default=50, help="Dimension of pretrained word vectors")
    command_parser.add_argument("-we", "--weight_embeddings",
        action="store_true", default=False,
        help="Whether to weight word embeddings as per Arora's paper.")
    command_parser.add_argument("-sw", "--include_stopwords", action="store_true",
        default=False, help="Include stopwords in data")
    # ------------------------ NN Architecture ------------------------
    command_parser.add_argument("-mhl", "--max_headline_len", type=int,
        default=None, help="maximum number of words per headline; if None, "
        "inferred from training data")
    command_parser.add_argument("-mbl", "--max_body_len", type=int,
        default=None, help="maximum number of words per body; if None, "
        "inferred from training data")
    command_parser.add_argument("-hd", "--hidden_sizes", type=int, nargs="+",
        default=[50], help="Dimensions of hidden represntations for each layer")
    command_parser.add_argument("-m", "--method", type=str,
        default="bag_of_words", help="Input embedding method; one of %s" %
        pprint.pformat(FNCModel.SUPPORTED_METHODS))
    command_parser.add_argument("-tm", "--transform_mean", action="store_true",
        default=False, help="Whether to further transform the mean in "
        "the bag_of_words model; if -tm is not supplied, hidden_sizes[-1] MUST "
        "equal input embedding dimension.")
    # ------------------------ Optimization Settings ------------------------
    command_parser.add_argument("-dp", "--dropout", type=float, default=0.0,
        help="Dropout probability")
    command_parser.add_argument("-r", "--regularizer", type=str, default=None,
        help="Regularizer to apply; one of l1 or l2")
    command_parser.add_argument("-p", "--penalty", type=float, default=1e-6,
        help="Regularization; ignored if regularizer is None")
    command_parser.add_argument("-ne", "--n_epochs", type=int, default=10,
        help="Number of training epochs.")
    command_parser.add_argument("-te", "--train_embeddings_epoch", type=int,
        default=11, help="Start training embeddings from this epoch onwards; "
        "embeddings are not trained if this argument is > n_epochs")
    command_parser.add_argument("-b", "--batch_size", type=int,
        default=52, help="The batch size")
    command_parser.add_argument("-ul", "--unweighted_loss",
        action="store_true", default=False,
        help="Include to use unweighted loss")
    command_parser.add_argument("-sm", "--similarity_metric",
        action="store_true", default=False,
        help="Train against a similarity loss function.")
    # ------------------------ Output Settings ------------------------
    command_parser.add_argument("-v", "--verbose", action="store_true",
        default=False)
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    layers = len(ARGS.hidden_sizes)
    embedding_path = "glove/glove.6B.%dd.txt" % ARGS.dimension

    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        config = Config(method=ARGS.method,
            embed_size=ARGS.dimension,
            hidden_sizes=ARGS.hidden_sizes,
            dropout=ARGS.dropout,
            transform_mean=ARGS.transform_mean,
            unweighted_loss=ARGS.unweighted_loss,
            similarity_metric=ARGS.similarity_metric,
            regularizer=ARGS.regularizer,
            penalty=ARGS.penalty,
            batch_size=ARGS.batch_size,
            n_epochs=ARGS.n_epochs,
            train_embeddings_epoch=ARGS.train_embeddings_epoch)

        handler = logging.FileHandler(config.log_output)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
           '%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
        logging.info("Arguments: %s", pprint.pformat(ARGS.__dict__))
        ARGS.func(train_bodies=ARGS.train_bodies,
            train_stances=ARGS.train_stances,
            dimension=ARGS.dimension,
            embedding_path=embedding_path, config=config,
            max_headline_len=ARGS.max_headline_len,
            max_body_len=ARGS.max_body_len,
            verbose=ARGS.verbose, 
            include_stopwords=ARGS.include_stopwords,
            weight_embeddings=ARGS.weight_embeddings)
