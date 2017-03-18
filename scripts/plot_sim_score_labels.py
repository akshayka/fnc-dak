import csv
import pickle

import matplotlib.pyplot as plt

TYPE = "binary_cosine"
TITLE_PREFIX = "Binary Cosine"
with open("%s_sim_scores.pickle" % TYPE, "rb") as f:
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
plt.title("%s Similarity Scores vs. Stances" % TITLE_PREFIX)
plt.xlabel("%s Similarity" % TITLE_PREFIX)
plt.savefig("%s_stances.png" % TYPE)

# See how well a quadratic regression in one variable would do
import cvxpy as cvx
import numpy as np

ss = np.array(ss)
q = cvx.Variable()
m = cvx.Variable()
b = cvx.Variable()
preds = q * (ss ** 2) + m * ss + b
obj = cvx.Minimize(cvx.norm(stances - preds, 1))
prob = cvx.Problem(obj)
prob.solve()

pred_vals = preds.value
pred_vals = np.round(pred_vals)
pred_vals = np.maximum(np.minimum(pred_vals, 3),  0)
unrelated_score = 0.25 * len(data_unrelated)
max_score = unrelated_score + 1.0 * (len(ss) - len(data_unrelated))
score = 0
for l, l_hat in zip(stances, pred_vals):
    if l == l_hat:
        score += 0.25
        if l != 0:
            score += 0.5
    if l in [1, 2, 3] and l_hat in [1, 2, 3]:
        score += 0.25
print "score %f" % (score / max_score)
print "unrelated score %f" % (unrelated_score / max_score)
