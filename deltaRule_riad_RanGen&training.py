import random as r
import numpy as np
import matplotlib.pyplot as mp

points = 10

def generate():
	classes = []
	data = []
	x = [r.random()*i/2 for i in range(0, points)]
	y = [i*0.5 for i in x]
	y[5:] = [i*0 + r.uniform(-1, 2) for i in range(0,5)]
	y[:5] = [i*0 - r.uniform(-1, 2) for i in range(0,5)]
	bias = [1 - 0 for i in range(0,points)]
	classes[5:] = [-1 - 0 for i in range(0,points)]
	classes[:5] = [1 - 0 for i in range(0,int(points/2))]
	data = np.array([list(a) for a in zip(x, y, bias, classes)])
	wts = np.array([i*0 + r.uniform(-1, 2) for i in range(0,3)])
	print('old:   ', wts)
	return data, classes, wts

data, classes, wts = generate()

def interact(ins, wts, t = 1):
	I = sum([i*w for i, w in zip(ins,wts)])
	observed = 1 if I > t else -1
	return observed

def screen(des, obs):
	if des == obs: return True
	else:
		return False

def delta(ins, wts, desired, observed, lr = 0.1):
	desired = ins[-1]
	new_weight = np.array([(((desired-observed)*lr)*i) + w for i, w in zip(ins,wts)])
	return new_weight

for i in range(len(data)):
	while(True):		
		obs = interact(data[i], wts)
		if screen(data[i][-1], obs):
			break
		else:
			x = data[i][0]
			y = data[i][1]
			bias = data[i][2]
			ins = [x, y, bias]
			wts = delta(ins, wts, data[i][-1], obs)
			#if you remove this break, it keeps going up
			break


print('result:', wts)
