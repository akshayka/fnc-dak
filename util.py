import os
import re
from collections import Counter
import pdb


import numpy as np


def read_stances(fstream):
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
		cleaned = re.findall(r"\w+|[^\w\s]", header)
		ret[0].append(cleaned)
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
			body += re.findall(r"\w+|[^\w\s]", cleaned)
			ret[body_id] = body
			body_id = None
			body = []
		else:
			body += re.findall(r"\w+|[^\w\s]", line.replace('\"\"', '\"'))
	return ret



def load_and_preprocess_fnc_data(args):
	bodies = read_bodies(args.train_bodies)
	stances = read_stances(args.train_stances)
	return bodies, stances
