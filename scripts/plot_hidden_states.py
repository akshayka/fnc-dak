import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

NORMALIZE = False
TYPE = "headlines"


PATH = "../pca/"
RESULTS_PATH = "../results/20170319_201712_300d_1L_vanilla_bag_of_words/results.txt"

gold = np.load(PATH + "stances_for_pca_run.npy")
preds = np.load(PATH + "preds_for_pca_run.npy")

h_h_reduced = np.load(PATH + "headline_hidden_pca.npy")
h_b_reduced = np.load(PATH + "body_hidden_pca.npy")

if NORMALIZE:
	headlines_norms = np.linalg.norm(h_h_reduced, axis=1)[:,np.newaxis]
	bodies_norms = np.linalg.norm(h_b_reduced, axis=1)[:,np.newaxis]

	h_h_reduced /= headlines_norms
	h_b_reduced /= bodies_norms

gold = np.load(PATH + "stances_for_pca_run.npy")
preds = np.load(PATH + "preds_for_pca_run.npy")

data = []

with open(RESULTS_PATH) as f:
	reader = csv.reader(f, delimiter="\t")

	for row in reader:
		if len(row) > 0:
			if TYPE == 'headlines':
				i = 0
			elif TYPE == 'bodies':
				i = 1
			data.append(row[i])

colors = ["b", "g", "r", "y", "c", "m", "w", "k"]
keywords = ["isis", "shooting", "murder", "apple", "job", "hair", "spider", "fire"]

indices = [[] for _ in xrange(len(keywords))]

for i, entry in enumerate(data):
	if gold[i] == preds[i]:
		for j, word in enumerate(keywords):
			if not entry.find(word) == -1:
				indices[j].append(i)
				break

def plot(inputs, color, label=None):
	x_coords = []
	y_coords = []
	for i in inputs:
		if TYPE == 'headlines':
			x_coords.append(h_h_reduced[i][0])
			y_coords.append(h_h_reduced[i][1])
		elif TYPE == 'bodies':
			x_coords.append(h_b_reduced[i][0])
			y_coords.append(h_b_reduced[i][1])

	plt.plot(x_coords, y_coords, color + "o", label=label)

plt.figure()

for i, word in enumerate(keywords):
	plot(indices[i], colors[i], label=word)

if NORMALIZE:
	plt.xlim((-1.1, 1.1))
	plt.ylim((-1.1, 1.1))

plt.legend()
plt.title("PCA of Select Hidden States")
plt.show()



