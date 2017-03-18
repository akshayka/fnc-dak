import csv
import pickle

import matplotlib.pyplot as plt

with open("sim_scores.pickle", "rb") as f:
    ss = pickle.load(f)

with open("../fnc-1-data/train_stances.csv", "rb") as f:
    f.readline() # skip header
    label_map = {"unrelated": 0, "discuss": 1, "disagree": 2, "agree": 3} 
    reader = csv.reader(f)
    stances = [label_map[row[2]] for row in reader]

data = zip(ss, stances)
data_unrelated = [(s, st) for s, st in data if st == 0]
data_discuss = [(s, st) for s, st in data if st == 1]
data_disagree = [(s, st) for s, st in data if st == 2]
data_agree = [(s, st) for s, st in data if st == 3]
plt.figure()

def plots(data, style, label):
    x, y = zip(*data)
    plt.plot(x, y, style, label=label)

plots(data_unrelated, "bo", "unrelated")
plots(data_discuss, "yo", "discuss")
plots(data_disagree, "ro", "disagree")
plots(data_agree, "go", "agree")
plt.yticks([0, 1, 2, 3])
ax = plt.gca()
ax.set_yticklabels(["unrelated", "discuss", "disagree", "agree"])
plt.title("Cosine Similarity Scores vs. Stances")
plt.xlabel("Cosine Similarity")
plt.savefig("cosine_stances.png")

