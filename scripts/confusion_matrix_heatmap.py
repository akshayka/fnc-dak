import numpy as np
import matplotlib.pyplot as plt

conf_arr = np.array([[7421, 175, 40, 5], 
            [50, 1604, 147, 62], 
            [1, 47, 135, 19], 
            [14, 124, 145, 403]])

norm_conf = []

temp_col_arr = []

for i in conf_arr.T:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    temp_col_arr.append(tmp_arr)
    # norm_conf.append(tmp_arr)



temp_col_arr = np.array(temp_col_arr)
norm_conf = (temp_col_arr.T).tolist()



fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width, height = conf_arr.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)

LABELS = ['unrelated', 'discuss', 'disagree', 'agree']
plt.xticks(range(width), LABELS)
plt.yticks(range(height), LABELS)
plt.xlabel('Guess')
plt.ylabel('Gold')
plt.title("Column-Normalized Confusion Matrix")
# plt.show()
plt.savefig('../plots/confusion_matrix_col.png', format='png')