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
from fnc import FNCModel, Config
from seq2seq import Seq2SeqModel


logger = logging.getLogger("baseline_model")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def do_train(train_bodies, train_stances, dimension, embedding_path, config, 
    max_headline_len=None, max_body_len=None, verbose=False, 
    include_stopwords=True, similarity_metric_feature=None, 
    weight_embeddings=False, idf=False, model="fnc"):
    logging.info("Loading training and dev data ...")
    fnc_data, fnc_data_train, fnc_data_dev = util.load_and_preprocess_fnc_data(
        train_bodies, train_stances, include_stopwords, 
        similarity_metric_feature)
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

    headlines_pc = bodies_pc = None
    if config.method == "arora":
        headlines_pc = util.arora_embeddings_pc(headline_vectors,
            embeddings)
        bodies_pc = util.arora_embeddings_pc(body_vectors,
            embeddings)
    else:
        headlines_pc = None
        bodies_pc = None

    if config.method == "vanilla_bag_of_words":
        logging.info("Precomputing training sentence embeddings ...")
        train_emb = embeddings
        if idf:
            train_emb = util.idf_embeddings(word_indices,
                headline_vectors + body_vectors, train_emb)
        headlines_emb = util.sentence_embeddings(headline_vectors, dimension,
            max_headline_len, train_emb)
        bodies_emb = util.sentence_embeddings(body_vectors, dimension,
            max_body_len, train_emb)
        training_data = [headlines_emb, bodies_emb, fnc_data_train.stances]
    else:
        training_data = [headline_vectors, body_vectors, fnc_data_train.stances]

    if similarity_metric_feature:
        training_data.append(fnc_data_train.sim_scores)
    training_data = zip(*training_data)

    # Vectorize and assemble the dev data; note that we use the training
    # maximum length
    dev_headline_vectors = util.vectorize(fnc_data_dev.headlines, word_indices,
        known_words, max_headline_len)
    dev_body_vectors = util.vectorize(fnc_data_dev.bodies, word_indices,
        known_words, max_body_len)

    if config.method == "vanilla_bag_of_words":
        logging.info("Precomputing dev sentence embeddings ...")
        test_emb = embeddings
        if idf:
            # TODO(akshayka): Experiment with using whole corpus as
            # documents vs just training vs just testing
            test_emb = util.idf_embeddings(word_indices,
                headline_vecotrs + dev_headline_vectors + body_vectors +
                dev_body_vectors, test_emb)
        dev_headlines_emb = util.sentence_embeddings(dev_headline_vectors,
            dimension, max_headline_len, test_emb)
        dev_bodies_emb = util.sentence_embeddings(dev_body_vectors,
            dimension, max_body_len, test_emb)
        dev_data = [dev_headlines_emb, dev_bodies_emb, fnc_data_dev.stances]
    else:
        dev_data = [dev_headline_vectors, dev_body_vectors,
            fnc_data_dev.stances]

    if similarity_metric_feature:
        dev_data.append(fnc_data_dev.sim_scores)
    dev_data = zip(*dev_data)

    with tf.Graph().as_default():
        logger.info("Building model %s ...", model)
        start = time.time()
        model_type = FNCModel if model == "fnc" else Seq2SeqModel
        model = model_type(config, max_headline_len, max_body_len, embeddings,
            headlines_pc=headlines_pc, bodies_pc=bodies_pc, verbose=verbose)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            logging.info('Fitting ...')
            model.fit(session, saver, training_data, dev_data)
            logging.info('Outputting ...')
            output = model.output(session, dev_data)

    indices_to_words = {word_indices[w] : w for w in word_indices}
    # TODO(akshayka): Please code-review this. In particular,
    # please validate whether dev_headline_vectors is an equivalent 
    # representation of output[0][0], and dev_body_vectors for output[0][1]
    headlines = [' '.join(
        util.word_indices_to_words(h, indices_to_words))
        for h in dev_headline_vectors]
    bodies = [' '.join(
        util.word_indices_to_words(b, indices_to_words))
        for b in dev_body_vectors]
    output = zip(headlines, bodies, output[1], output[2])

    with open(model.config.eval_output, 'w') as f, open(
        model.config.error_output, "w") as g:
        for headline, body, label, prediction in output:
            f.write("%s\t%s\tgold:%d\tpred:%d\n\n" % (
                headline, body, label, prediction))
            if label != prediction:
                g.write("%s\t%s\tgold:%d\tpred:%d\n\n" % (
                    headline, body, label, prediction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains and tests FNCModel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ------------------------ Input Data ------------------------
    parser.add_argument("-tb", "--train_bodies",
        type=argparse.FileType("r"), default="fnc-1-data/train_bodies.csv",
        help="Training data")
    parser.add_argument("-ts",
        "--train_stances", type=argparse.FileType("r"),
        default="fnc-1-data/train_stances.csv", help="Training data")
    parser.add_argument("-d", "--dimension", type=int,
        default=50, help="Dimension of pretrained word vectors")
    parser.add_argument("-we", "--weight_embeddings",
        action="store_true", default=False,
        help="Whether to weight word embeddings as per Arora's paper.")
    parser.add_argument("-idf", action="store_true",
        default=False, help="Whether to weight word embeddings with idf.")
    parser.add_argument("-sw", "--include_stopwords", action="store_true",
        default=False, help="Include stopwords in data")
    parser.add_argument("-smf", "--similarity_metric_feature", type=str,
        default=None, help="Type of similarity metric features to add; "
        "one of %s" % pprint.pformat(
        FNCModel.SUPPORTED_SIMILARITY_METRIC_FEATS))
    # ------------------------ NN Architecture ------------------------
    parser.add_argument("-mo", "--model", type=str,
        default="fnc", help="NN model; either fnc or seq2seq")
    parser.add_argument("-mhl", "--max_headline_len", type=int,
        default=None, help="maximum number of words per headline; if None, "
        "inferred from training data")
    parser.add_argument("-mbl", "--max_body_len", type=int,
        default=None, help="maximum number of words per body; if None, "
        "inferred from training data")
    parser.add_argument("-hd", "--hidden_sizes", type=int, nargs="+",
        default=[50], help="Dimensions of hidden represntations for each layer")
    parser.add_argument("-m", "--method", type=str,
        default="bag_of_words", help="Input embedding method; one of %s" %
        pprint.pformat(FNCModel.SUPPORTED_METHODS))
    parser.add_argument("-tm", "--transform_mean", action="store_true",
        default=False, help="Whether to further transform the mean in "
        "the bag_of_words model; if -tm is not supplied, hidden_sizes[-1] MUST "
        "equal input embedding dimension.")
    # ------------------------ Optimization Settings ------------------------
    parser.add_argument("-dp", "--dropout", type=float, default=0.0,
        help="Dropout probability")
    parser.add_argument("-r", "--regularizer", type=str, default=None,
        help="Regularizer to apply; one of l1 or l2")
    parser.add_argument("-p", "--penalty", type=float, default=1e-6,
        help="Regularization; ignored if regularizer is None")
    parser.add_argument("-ne", "--n_epochs", type=int, default=10,
        help="Number of training epochs.")
    parser.add_argument("-te", "--train_embeddings_epoch", type=int,
        default=None, help="Start training embeddings from this epoch onwards; "
        "embeddings are not trained if this argument is None or > n_epochs")
    parser.add_argument("-b", "--batch_size", type=int,
        default=52, help="The batch size")
    parser.add_argument("-ul", "--unweighted_loss",
        action="store_true", default=False,
        help="Include to use unweighted loss")
    parser.add_argument("-sm", "--scoring_metrics", type=str, nargs="+",
        default=None, help="Train by regressing a similarity "
        "score against the labels; the leading arguments must be a subset of "
        "%s, while the last argument must be a positive number specifying the "
        "the (polynomial) degree of the regression. "% pprint.pformat(
        FNCModel.SUPPORTED_SCORING_METRICS))
    # ------------------------ Output Settings ------------------------
    parser.add_argument("-v", "--verbose", action="store_true",
        default=False)

    ARGS = parser.parse_args()

    # Argument validation
    layers = len(ARGS.hidden_sizes)
    embedding_path = "glove/glove.6B.%dd.txt" % ARGS.dimension

    assert ARGS.method in FNCModel.SUPPORTED_METHODS
    if ARGS.method == "arora":
        ARGS.weight_embeddings = True
    if ARGS.method == "vanilla_bag_of_words" and \
        ARGS.train_embeddings_epoch is not None and \
        ARGS.train_embeddings_epoch <= ARGS.n_epochs:
        logging.fatal("Embeddings cannot be trained in the "
            "vanilla_bag_of_words model.")
        sys.exit(1)
    if ARGS.method == "vanilla_bag_of_words" and len(ARGS.hidden_sizes) > 1:
        logging.fatal("Multiple layer networks are not yet supported for "
            "vanilla_bag_of_words.")
        sys.exit(1)

    config = Config(method=ARGS.method,
        embed_size=ARGS.dimension,
        hidden_sizes=ARGS.hidden_sizes,
        dropout=ARGS.dropout,
        transform_mean=ARGS.transform_mean,
        unweighted_loss=ARGS.unweighted_loss,
        scoring_metrics=ARGS.scoring_metrics,
        regularizer=ARGS.regularizer,
        penalty=ARGS.penalty,
        batch_size=ARGS.batch_size,
        n_epochs=ARGS.n_epochs,
        similarity_metric_feature=ARGS.similarity_metric_feature,
        train_embeddings_epoch=ARGS.train_embeddings_epoch)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
       '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.info("Arguments: %s", pprint.pformat(ARGS.__dict__))
    do_train(train_bodies=ARGS.train_bodies,
        train_stances=ARGS.train_stances,
        dimension=ARGS.dimension,
        embedding_path=embedding_path, config=config,
        max_headline_len=ARGS.max_headline_len,
        max_body_len=ARGS.max_body_len,
        verbose=ARGS.verbose, 
        include_stopwords=ARGS.include_stopwords,
        similarity_metric_feature=ARGS.similarity_metric_feature,
        weight_embeddings=ARGS.weight_embeddings,
        idf=ARGS.idf,
        model=ARGS.model)
