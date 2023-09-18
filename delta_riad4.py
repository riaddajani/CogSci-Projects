import numpy as np
import random as r
import matplotlib.pyplot as mp

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

theta = 5
a = 0.2
points = 10
bias = 1
spread = 10
seperation = 3

# temp
m = 0.5
b = 5
cl = []
b = [1]

def generate_data():
        x = np.array([r.random() * spread/2 for i in range(0,points)])
        y = x*m+b
        split1 = y[:int(points/2)] + r.random()*seperation
        split2 = y[int(points/2):] - r.random()*seperation
        y = np.append(split1,split2)
        for i in range(0, points):
            cl[:5] = np.asarray([1 - 0 for i in range(0,points)])
            cl[5:] = np.asarray([-1 - 0 for i in range(0,points)])
        coords = np.asarray(list(zip(x, y)))
        data = np.asarray(list(zip(coords, cl)))
        # for i in range(0,points):
        #     self.d = np.asarray(list(zip(coords, self.c)))
        wts = np.asarray([r.random() for i in range(2)])
        return data, wts, x, y, cl

def deltaRule(data, wts):
	des = [data[1][-1] for i in b]
	I = sum([i[0]*wt for i,wt in zip(data[0:-1],wts)])
	obs = 1 if I.all() > theta else -1
	new_wts = [wts + (a*(obs-des[0]))*i for i in b]
	print('old', wts)
	print('new', new_wts)
	
	if obs != des:
		new = np.dot(data[1][-1], new_wts)
		print(new)
	


data, wts, x, y, cl = generate_data()
deltaRule(data, wts)

colors = []
for i in range(len(x)):
    if x[i] > y[i]:
        colors.append('blue')
    else:
        colors.append('red')

mp.scatter(x, y, c=colors)
mp.show()