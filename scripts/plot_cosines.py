import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

PATH = "../pca/"

cosine = np.load(PATH + "cosine_pca.npy")

gold = np.load(PATH + "stances_for_pca_run.npy")
preds = np.load(PATH + "preds_for_pca_run.npy")

i_agree = []
i_disagree = []
i_discuss = []
i_unrelated = []
k_agree = []
k_disagree = []
k_discuss = []
k_unrelated = []

labels = zip(gold, preds)
for i, label in enumerate(labels):
	if not label[0] == label[1]:
		if label[0] == 3:
			k_agree.append(i)
		elif label[0] == 2:
			k_disagree.append(i)
		elif label[0] == 1:
			k_discuss.append(i)
		else:
			k_unrelated.append(i)
	if label[1] == 3:
		i_agree.append(i)
	elif label[1] == 2:
		i_disagree.append(i)
	elif label[1] == 1:
		i_discuss.append(i)
	else:
		i_unrelated.append(i)

# indices = [0, 1, 16, 17]
# labels = ["disagree", "unrelated", "agree", "discuss"]

# data = zip(indices, labels)

"""0 is 2, spider burrowed tourists stomach chest  fear story spiderman might seemed perth scientists cast doubt
 claims spider burrowed mans body first trip bali story went global thursday generating hundreds stor
ies online earlier month dylan thomas headed holiday island sought medical help experiencing really b
urning sensation like searing feeling abdomen dylan thomas says spider crawl underneath skin thomas s
aid specialist dermatologist called later used tweezers remove believed tropical spider seems may cau
ght web misinformation arachnologist dr volker said whatever creature almost impossible culprit spide
r look spider fangs mouth parts able burrow cant get skin said thought may something like mite differ
ent parasitic mites sometimes look bit like  gold:2  pred:2

	1 is 0, identity isis terrorist known jihadi john reportedly revealed   adding apples ios launch troubles rep
ort monday claims operating systems reset settings feature hardware wiping user preferences also dele
tes data stored icloud drive discovered issue presents iphone ipad users attempt reset preferences vi
a devices settings general reset reset settings option feature allows users troubleshoot possible ios
 issues reverting system settings back factory baselines os yosemite subsequently icloud drive yet av
ailable consumer download aside apples public beta testing program effects consumers non existent how
ever large number untrained public beta participants opting service ios problem could become serious 
according report one ipad user performed reset found local iwork documents deleted consequently wiped
        gold:0  pred:0

	16 is 3, india rape crisis sees mob castrate alleged sex attacker shocking scenes        alleged attempted rapist india received grisly punishment alleged attack young teenager angry mob dragged butcher shop penis severed meat cleaver suresh kumar set upon locals city ganganagar india north western rajasthan state heard girl screams mob dragged kumar away found alley allegedly pinning victim wall group held vigilante community meeting decide beaten sticks hour dragged local butcher shop organ severed meat cleaver thrown street kumar still critical condition attack local man aamir dhawan said one went help man could see penis ground knew punishment sex crime masked gang filmed smashing jewellers daring daylight robbery lot intolerable offences women country recently  gold:3  pred:3

	17 is 1, nigeria hopes return kidnapped schoolgirls rise ceasefire reported      girls kidnapped nigeria released country government agreed immediate cease fire captors boko haram air marshal alex chief defence staff ordered troops immediately comply agreement ceasefire agreement concluded federal government nigeria jama atu sunna wal jihad boko haram said brave shameless pr stunt nigerian singer offers virginity boko haram exchange kidnapped schoolgirls news comes another official confirmed direct negotiations week neighboring chad release girls taken april initial students kidnapped boarding school northeast town number already managed escape boko haram demanding release detained extremists exchange girls military force option rescue girls kidnapped boko haram nigeria     gold:1  pred:1 """

def plot(inputs, color, label=None, length=1):
	x_coords = []
	y_coords = []
	for i in inputs:
		x = length * cosine[i]
		y = np.sqrt(np.square(length) - np.square(x))

		x_coords.append(x)
		y_coords.append(y)

	plt.plot(x_coords, y_coords, color + "o", label=label)

plt.figure()

plot(i_agree, "r", label="agree")
plot(i_disagree, "b", label="disagree", length=0.98)
plot(i_discuss, "g", label="discuss")
plot(i_unrelated, "y", label="unrelated", length=0.98)
plot(k_discuss, "g", length=0.8)
plot(k_unrelated, "y", length=0.75)
plot(k_agree, "r", length=0.7)
plot(k_disagree, "b", length=0.65)


plt.xlim((0, 1.1))
plt.ylim((0, 1.1))

plt.legend()
plt.title("Mapping from Angles to Predictions")
plt.show()



