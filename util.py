from collections import Counter, defaultdict, namedtuple
import csv
import logging
import os
import pickle
import re
import time
import random
import sys

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

# TODO(akshayka): Add a field for cosine similarity
FNCData = namedtuple("FNCData", ["headlines", "bodies", "stances", "sim_scores",
    "max_headline_len", "max_body_len"])
# TODO(akshayka): What about special punctuation like "@" or "#"?
TOKEN_RE = r"[a-zA-Z]+[']?[a-zA-Z]+"

LBLS = ["unrelated", "discuss", "disagree", "agree"]

UNRELATED = 0
DISCUSS = 1
DISAGREE = 2
AGREE = 3

RELATED = [DISCUSS, DISAGREE, AGREE]

STOPWORDS = set(stopwords.words('english'))



# ----------------- Utilities for evaluation ------------------
def to_table(data, row_labels, column_labels, precision=2, digits=4):
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    data = [["%04.2f"%v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret


class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        """
        Attributes:
            labels : list of str
                List of all possible labels. labels[i] must correspond to
                numeric prediction i.
            default_label : int
        """
        self.labels = labels
        self.default_label = default_label if default_label is not None else \
            len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] \
            for l,_ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = np.array([0., 0., 0., 0.])
        micro = np.array([0., 0., 0., 0.])
        default = np.array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in 
                keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0  else 0

            # update micro/macro averages
            micro += np.array([tp, fp, tn, fn])
            macro += np.array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not
                                        # the default label!
                default += np.array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return to_table(data, self.labels + ["micro","macro","not-O"], ["label",
            "acc", "prec", "rec", "f1"])

# -------------- Utilities for running epochs ---------------
class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), (current
                     - self.seen_so_far)]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, 
                        self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, 
                        self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


# TODO(akshayka): Balanced minibatches
def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use 
    this function to iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, 
            labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of 
            each element in the list. This can be used to iterate through 
            multiple data sources (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) 
        is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:(minibatch_start + 
            minibatch_size)]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] \
        for i in minibatch_idx]


def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


# ---------------- Utilities for data processing -------------
PAD_TOKEN = "___PPPADDD___"

def sentence_embeddings(examples, dimension, max_len, embeddings):
    emb = tf.constant(embeddings, dtype=tf.float32)
    x = tf.nn.embedding_lookup(emb, examples)
    x = tf.reshape(x, (-1, max_len, dimension))
    used = tf.sign(tf.reduce_max(tf.abs(x), axis=2))
    seqlen = tf.cast(tf.reduce_sum(used, axis=1), tf.int32)
    seqlen_scale = tf.cast(tf.expand_dims(seqlen, axis=1), tf.float32)
    # weighted sentence embeddings (without removal of PC)
    X = tf.divide(tf.reduce_sum(input_tensor=x, axis=1), seqlen_scale)
    with tf.Session() as sess:
        X = sess.run(X)
    return X

def countFeaturizer(binary_counts, vocab=None):
    stemmer = PorterStemmer()
    analyzer = CountVectorizer(stop_words="english").build_analyzer()

    def stemWords(text):
        return (stemmer.stem(x) for x in analyzer(text))

    stemmed_vectorizer = CountVectorizer(analyzer=stemWords, 
        binary=binary_counts, vocabulary=vocab)
    return stemmed_vectorizer

def similarity_metrics(vectorizer, headlines, bodies, similarity_metric_feature):
    assert similarity_metric_feature is not None
    logging.info("Creating similarity metric features: %s ..." 
        % similarity_metric_feature)

    print "len headlines: %d, len bodies: %d" % (len(headlines), len(bodies))

    featurized_bodies = vectorizer.transform(headlines)
    featurized_headlines = vectorizer.transform(bodies)

    # TODO(delenn): implement tfidf weighting?
    # if tfidf:
    #     transformer = TfidfTransformer()
    #     tfidf_headlines = transformer.fit_transform(featurized_headlines)
    #     tfidf_bodies = transformer.fit_transform(featurized_bodies)

    assert np.shape(featurized_bodies)[1] == np.shape(featurized_headlines)[1]

    if similarity_metric_feature == "jaccard":
        sim_scores = [jaccard_similarity_score(featurized_headlines[i], 
            featurized_bodies[i]) for i in range(len(headlines))]


        # sim_scores = []
        # for i in range(len(headlines)):
        #     sim_scores.append(jaccard_similarity_score(featurized_headlines[i]))
        print "sim_scores length: %d" % len(sim_scores)
    else: # Use cosine similarity
        sim_scores = np.diagonal(cosine_similarity(featurized_headlines, 
            featurized_bodies))

    logging.info("Finished similarity metric features ...")
    return sim_scores


def arora_embeddings_pc(vectorized_examples, embeddings):
    unique_inputs = [list(i) for i in set(map(tuple, vectorized_examples))]
    # very hacky code ...
    emb = tf.constant(embeddings, dtype=tf.float32)
    x = tf.nn.embedding_lookup(emb, unique_inputs)
    used = tf.sign(tf.reduce_max(tf.abs(x), axis=2))
    seqlen = tf.cast(tf.reduce_sum(used, axis=1), tf.int32)
    seqlen_scale = tf.cast(tf.expand_dims(seqlen, axis=1), tf.float32)
    # weighted sentence embeddings (without removal of PC)
    X = tf.divide(tf.reduce_sum(input_tensor=x, axis=1), seqlen_scale)
    with tf.Session() as sess:
        X = sess.run(X)
        
    # TODO(akshayka): should X be centered?
    X = X.reshape(X.shape[0], X.shape[2])
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    return pc.T # shape (embedding dimension, 1)

    
# Taken from Arora's code: https://github.com/YingyuLiang/SIF/
def get_word_weights(weightfile, a=1e-3):
    if a <=0: # when the parameter makes no sense, use unweighted
       raise ValueError("a <= 0!")

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.iteritems():
        word2weight[key] = a / (a + value/N)
    return word2weight


def word_indices_to_words(example, indices_to_words):
    # NB: 0 is the padding index
    # TODO(akshayka): HACK -- i is a list of features
    return [indices_to_words[i[0]] for i in example if i[0] != 0]


def vectorize(examples, word_indices, known_words, max_len):
    """Convert a list of examples with word tokens to word indices.

    Args:
        examples: list of lists, each sublist is one example
            (i.e., a list of words)
        word_indices: dict word -> index
        known_words: set : words in word_indices that are also in glove
        max_len: maximum length of any example
    Returns:
        vectorized_examples: list of lists, each sublist is one example and
            each entry in sublist corresponds to an index in the embedding
            matrix
    """
    pad_idx = word_indices[PAD_TOKEN]
    examples = [[w for w in e if w in known_words] for e in examples]
    vectorized_examples = [(
        [(word_indices[w],) for w in e] + \
        [(pad_idx,)] * max(max_len - len(e), 0))[:max_len] for e in examples]
    return vectorized_examples


def compute_idfs(indices_to_words, pad_idx, examples):
    logging.info("Computing document frequencies ...")
    df_counts = defaultdict(int)
    for e in examples:
        words_in_example = set([i[0] for i in e if i[0] != pad_idx])
        for w_i in words_in_example:
            df_counts[w_i] += 1

    logging.info("Computing idfs ...")
    num_docs = len(examples)
    idfs = {}
    for w_i,c in df_counts.iteritems():
       idfs[w_i] = np.log(num_docs / (1 + c))
    return idfs


def idf_embeddings(word_indices, examples, embeddings):
    pad_idx = word_indices[PAD_TOKEN]
    idfs = compute_idfs(word_indices, pad_idx, examples)
    idf_embeddings = np.copy(embeddings)
    for w_i,idf  in idfs.iteritems():
        idf_embeddings[w_i] *= idf
    return idf_embeddings
    

def load_embeddings(word_indices, dimension=300,
    embedding_path="glove/glove.6B.300d.txt", weight_embeddings=False):
    embeddings = np.zeros([len(word_indices) + 1, dimension])
    glove_words = set([])
    weights = None if not weight_embeddings else \
        get_word_weights("aux_data/enwiki_vocab_min200.txt")
    words_without_weights = 0
    with open(embedding_path, 'rb') as fstream:
        for line in fstream:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            word = row[0]
            if word not in word_indices:
                continue
            data = np.asarray([float(x) for x in row[1:]])
            if weight_embeddings:
                if word in weights:
                    data = weights[word] * data
                else:
                    words_without_weights += 1
            if len(data) != dimension:
                raise RuntimeError("wrong number of dimensions; "
                    "expected %d, saw %d" % (dimension, len(data)))
            # TODO(akshayka): if using arora's embeddings, multiply 
            # each embedding by its word weight
            embeddings[word_indices[word]] = data
            glove_words.add(word)
    our_words = set(word_indices.keys())
    # TODO(delenn): account for unseen words (unk token?)
    unk = our_words.difference(glove_words)
    if len(unk) > 0:
        logging.warning("%d unknown words out of %d total", len(unk),
            len(word_indices))
    if words_without_weights > 0:
        logging.warning("%d words do not have weights", words_without_weights)
    known_words = our_words.intersection(glove_words)
    return embeddings, known_words
        

def process_corpus(corpus):
    """Return dict : word -> index"""
    tokens = {}
    # NB: idx 0 is reserved for the padding vector
    tokens[PAD_TOKEN] = 0
    idx = 1
    for word in corpus:
        if word not in tokens: 
            tokens[word] = idx
            idx += 1
    return tokens


def tokenize_text(text, include_stopwords):
    if include_stopwords:
        tokenized = [tok.replace("'", "") for tok in re.findall(TOKEN_RE, 
            text.lower())]
    else:
        tokenized = [tok.replace("'", "") for tok in re.findall(TOKEN_RE, 
            text.lower()) if tok not in STOPWORDS]
    return tokenized

def read_stances(fstream, include_stopwords):
    """Returns headlines, ids, stances"""
    fstream.readline() # read past the header
    csv_reader = csv.reader(fstream)
    headlines = []
    body_ids = []
    stances = []
    for row in csv_reader:
        headlines.append(tokenize_text(row[0], include_stopwords))
        body_ids.append(int(row[1]))
        stance = row[2]
        if stance == "unrelated":
            stances.append(UNRELATED)
        elif stance == "discuss":
            stances.append(DISCUSS)
        elif stance == "disagree":
            stances.append(DISAGREE)
        elif stance == "agree":
            stances.append(AGREE)
        else:
            raise ValueError("Unknown stance %s" % stance)
    return [headlines, body_ids, stances]


def read_bodies(fstream, include_stopwords):
    """ 
    Basic pre-processing for bodies. Bodies can span multiple lines, beginning
    and ends denoted by quotes. Quotes within a body are denoted with double
    quotes (""). Each body ends with "\n, so we take off the last two
    characters when reading them in.
    """
    fstream.readline() # read past the header
    csv_reader = csv.reader(fstream)
    body_map = {int(row[0]) : tokenize_text(row[1], 
        include_stopwords) for row in csv_reader} 
    return body_map


def load_and_preprocess_fnc_data(train_bodies_fstream, train_stances_fstream, 
    include_stopwords=False, similarity_metric_feature=None,
    train_test_split=0.8):
    stances = read_stances(train_stances_fstream, include_stopwords)
    body_ids = stances[1]
    unique_body_ids = list(set(body_ids))
    random.shuffle(unique_body_ids)

    # Ensure that the set of bodies present in the training data is
    # disjoint from the set of bodies present in the test data
    num_examples = len(stances[0])
    split = int(train_test_split * len(unique_body_ids))
    train_body_ids = set(unique_body_ids[:split])
    train_indices = [i for i, body_id in \
        enumerate(body_ids) if body_id in train_body_ids]
    set_train = set(train_indices)
    test_indices = [i for i in range(num_examples) if i not in set_train]
    assert len(set_train.intersection(set(test_indices))) == 0

    # Overwrite body_ids with the body text
    body_map = read_bodies(train_bodies_fstream, include_stopwords)
    text_bodies = [body_map[body_id] for body_id in body_ids]
    stances[1] = text_bodies
    fnc_data = FNCData(
        headlines=stances[0], bodies=stances[1], stances=stances[2],
        sim_scores=None, max_headline_len=0, max_body_len=0)

    # Populate training data from train_indices
    train_headlines = [fnc_data.headlines[i] for i in train_indices]
    train_bodies = [fnc_data.bodies[i] for i in train_indices]
    train_sim_scores = None
    if similarity_metric_feature is not None:
        string_headlines = [" ".join(h) for h in train_headlines]
        string_bodies = [" ".join(b) for b in train_bodies]
        binary_counts = True if similarity_metric_feature == "jaccard" else False
        train_vectorizer = countFeaturizer(binary_counts)
        train_vectorizer = train_vectorizer.fit(string_headlines + string_bodies)
        train_vocab = train_vectorizer.vocabulary_
        train_sim_scores = similarity_metrics(train_vectorizer, string_headlines, 
            string_bodies, similarity_metric_feature)


    headline_lens = [len(head) for head in train_headlines]
    max_headline_len = max(headline_lens)
    body_lens = [len(body) for body in train_bodies]
    max_body_len = max(body_lens)
    fnc_data_train = FNCData(
        headlines=train_headlines, bodies=train_bodies,
        stances=[fnc_data.stances[i] for i in train_indices],
        sim_scores=train_sim_scores,
        max_headline_len=max_headline_len, max_body_len=max_body_len)

    # Populate test data from test_indices
    test_headlines = [fnc_data.headlines[i] for i in test_indices]
    test_bodies = [fnc_data.bodies[i] for i in test_indices]
    test_sim_scores = None
    if similarity_metric_feature is not None:
        string_headlines = [" ".join(h) for h in test_headlines]
        string_bodies = [" ".join(b) for b in test_bodies]
        test_vectorizer = countFeaturizer(binary_counts, vocab=train_vocab)
        test_sim_scores = similarity_metrics(test_vectorizer, string_headlines, 
            string_bodies, similarity_metric_feature)

    fnc_data_test = FNCData(
        headlines=test_headlines,
        bodies=test_bodies,
        stances=[fnc_data.stances[i] for i in test_indices],
        sim_scores=test_sim_scores,
        max_headline_len=max_headline_len, max_body_len=max_body_len)

    return fnc_data, fnc_data_train, fnc_data_test
