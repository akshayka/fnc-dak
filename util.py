from collections import Counter, defaultdict, namedtuple
import csv
import logging
import os
import pickle
import re
import time
import sys

import numpy as np

FNCData = namedtuple("FNCData", ["headlines", "bodies", "stances",
    "max_headline_len", "max_body_len"])
TOKEN_RE = r"\w+[']?[\w+]?"

LBLS = ["unrelated", "discuss", "disagree", "agree"]

UNRELATED = 0
DISCUSS = 1
DISAGREE = 2
AGREE = 3

RELATED = [DISCUSS, DISAGREE, AGREE]



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
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

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
        return to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])

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
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
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
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
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
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
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
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def minibatches(data, batch_size, shuffle=True):
    batches = [np.array(col) for col in zip(*data)]
    return get_minibatches(batches, batch_size, shuffle)


# ---------------- Utilities for data processing -------------
PAD_TOKEN = "___PPPADDD___"


def word_indices_to_words(example, indices_to_words):
    # NB: 0 is the padding index
    # TODO(akshayka): HACK -- i is a list of features
    return [indices_to_words[i[0]] for i in example if i[0] != 0]


def vectorize(examples, word_indices, max_len):
    """Convert a list of examples with word tokens to word indices.

    Args:
        examples: list of lists, each sublist is one example
            (i.e., a list of words)
        word_indices: dict : word -> index
        max_len: maximum length of any example
    Returns:
        vectorized_examples: list of lists, each sublist is one example and
            each entry in sublist corresponds to an index in the embedding
            matrix
    """
    pad_idx = word_indices[PAD_TOKEN]
    vectorized_examples = [([[word_indices[w]] for w in e] + \
        [[pad_idx]] * max(max_len - len(e), 0))[:max_len] for e in examples]
    return vectorized_examples


def load_embeddings(word_indices, dimension=300,
    embedding_path="glove/glove.6B.300d.txt"):
    embeddings = np.zeros([len(word_indices) + 1, dimension])
    with open(embedding_path, 'rb') as fstream:
        for line in fstream:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            word = row[0]
            if word not in word_indices:
                # TODO(delenn): account for unseen words (unk token?)
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimension:
                raise RuntimeError("wrong number of dimensions; "
                    "expected %d, saw %d" % (dimension, len(data)))
            embeddings[word_indices[word]] = np.asarray(data)
    return embeddings
        

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


def tokenize_text(text):
    return [tok.replace("'", "") for tok in re.findall(TOKEN_RE, text.lower())]


def read_stances(fstream):
    """Returns headlines, ids, stances"""
    fstream.readline() # read past the header
    csv_reader = csv.reader(fstream)
    headlines = []
    body_ids = []
    stances = []
    for row in csv_reader:
        headlines.append(tokenize_text(row[0]))
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


def read_bodies(fstream):
    """ 
    Basic pre-processing for bodies. Bodies can span multiple lines, beginning
    and ends denoted by quotes. Quotes within a body are denoted with double
    quotes (""). Each body ends with "\n, so we take off the last two
    characters when reading them in.
    """
    fstream.readline() # read past the header
    csv_reader = csv.reader(fstream)
    body_map = {int(row[0]) : row[1] for row in csv_reader} 
    for body_id in body_map:
        body_map[body_id] = tokenize_text(body_map[body_id])
    return body_map


def load_and_preprocess_fnc_data(train_bodies_fstream, train_stances_fstream,
    train_test_split=0.7):
    bodies = read_bodies(train_bodies_fstream)
    stances = read_stances(train_stances_fstream)
    body_ids = stances[1]
    for i, body_id in enumerate(body_ids):
        body_ids[i] = bodies[body_id]
    fnc_data = FNCData(
        headlines=stances[0], bodies=stances[1], stances=stances[2],
        max_headline_len=0, max_body_len=0)

    # Populate training data
    num_examples = len(fnc_data.headlines)
    train_indices = np.random.choice(np.arange(num_examples), 
        size=int(train_test_split * num_examples), replace=False)

    train_headlines = [fnc_data.headlines[idx] for idx in train_indices]
    headline_lens = [len(head) for head in train_headlines]
    max_headline_len = max(headline_lens)

    train_bodies = [fnc_data.bodies[idx] for idx in train_indices]
    body_lens = [len(body) for body in train_bodies]
    max_body_len = max(body_lens)

    fnc_data_train = FNCData(
        headlines=train_headlines, bodies=train_bodies,
        stances=[fnc_data.stances[idx] for idx in train_indices],
        max_headline_len=max_headline_len, max_body_len=max_body_len)

    # Populate test data
    test_indices = [idx for idx in np.arange(num_examples) \
        if idx not in train_indices]
    fnc_data_test = FNCData(
        headlines=[fnc_data.headlines[idx] for idx in test_indices],
        bodies=[fnc_data.bodies[idx] for idx in test_indices],
        stances=[fnc_data.stances[idx] for idx in test_indices],
        max_headline_len=max_headline_len, max_body_len=max_body_len)

    return fnc_data, fnc_data_train, fnc_data_test
