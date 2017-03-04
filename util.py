import os
import re
from collections import Counter
import pdb


import numpy as np


def load_embeddings(word_indices, dimension=300, embedding_path):
    embeddings = np.zeros([len(word_indices)], dimension)
    with open(embedding_path, 'rb') as fstream:
        for line in fstream:
            line = line.strip()
            if not line:
                continue
            row = line.split()
            word = row[0]
            if word not in word_indices:
                continue
            data = [float(x) for x in row[1:]]
            if len(data) != dimensions:
                raise RuntimeError("wrong number of dimensions; "
                    "expected %d, saw %d" % (dimension, len(data)))
            embeddings[word_indices[word]] = np.asarray(data)
    return embeddings
        

def process_corpus(corpus):
    tokens = {}
    idx = 0
    for word in corpus:
        if word not in tokens: 
            tokens[word] = idx
            idx += 1
    return tokens


def read_stances(fstream):
    '''Returns headlines, ids, stances'''
    ret = [[], [], []]
    fstream.readline()
    for line in fstream:
        first = line.rfind(',')
        second = line[:first].rfind(',')
        header = line[:second]
        # clean out extra quotes for headers with commas
        if header[0] == '\"' and header[-1] == '\"':
            header = header[1:-1]
        # split all punctuation, paranthesis, quotes, etc. from words
        # cleaned = re.findall(r"\w+|[^\w\s]", header)
        cleaned = re.findall(r"\w+", header)
        ret[0].append([w.lower() for w in cleaned])
        ret[1].append(int(line[second+1:first]))
        ret[2].append(line[first+1:])
    return ret      


def read_bodies(fstream):
    """ 
    Basic pre-processing for bodies. Bodies can span multiple lines, beginning
    and ends denoted by quotes. Quotes within a body are denoted with double
    quotes (""). Each body ends with "\n, so we take off the last two
    characters when reading them in.
    """
    ret = {}
    fstream.readline()
    body_id = None
    body = []
    for line in fstream:
        if body_id is None:
            first = line.find(',')
            body_id = int(line[:first])
            line = line[first +2:]
        # check if line is last in body
        if (len(line) > 5 and line[-2] == '\"' and
            (line[-3] != '\"' or line[-4:-1] == '\"\"\"')):
            cleaned = line[:-2].replace('\"\"', '\"')
            # body += re.findall(r"\w+|[^\w\s]", cleaned)
            body += re.findall(r"\w+", cleaned)
            ret[body_id] = [w.lower() for w in body]
            body_id = None
            body = []
        else:
            body += re.findall(r"\w+|[^\w\s]", line.replace('\"\"', '\"'))
    return ret



def load_and_preprocess_fnc_data(train_bodies, train_stances):
    bodies = read_bodies(train_bodies)
    stances = read_stances(train_stances)
    return bodies, stances
