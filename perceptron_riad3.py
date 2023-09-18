import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

theta = 1
ins = np.array([
    [np.array([0.3,0.7])   ,  1.0],
    [np.array([-0.6,0.3])  , -1.0],
    [np.array([0.7,0.3])   ,  1.0],
    [np.array([-0.2,-0.8]) , -1.0]])

wts = np.array([-0.6, 0.8])

new_data = []
computed_class = []
correctness = []

def compute_activation(data, weights):
    for i in range(len(data)):
        calc = data[i][0]
        for j in range(len(calc)):
            I = np.add(calc[j], weights[j]) # Printing I will show -0.3, 1.5 (expected values)
            new_data.append(I)
            if I > theta:
                y = 1.0
                computed_class.append(y)
            else:
                y = -1.0
                computed_class.append(y)
    return computed_class

def compare_class(computed, desired):
    for i in range(len(computed)):
        for j in range(len(desired)):
            if computed[i] == desired[j][1]: # This is wrong because it neglects the last 4 values
                beta = 1.0 
            else: # (x, y) if input x > T then y = 1, if input y > T then y = 1 (for AND since (1, 1) x & y = 1)
                beta = -1.0
        correctness.append(beta)
    return correctness


def update_wts(data, weights):
    computed_class = compute_activation(data, weights) # The inputs go through the functions more than once. 'I' shows one iteration.
    correctness = compare_class(computed_class, data)
    updt_wts = []
    for i in range(len(correctness)):
        by = correctness[i]*computed_class[i]
        for j in range(len(data)):
            for k in range(len(weights)):
                byx = by*data[j][0][k]
                Wnew = weights[k]+byx
                data[j][0][k] = Wnew # Because of the previous error, the new classes arent updated.
            return data

data = update_wts(ins, wts)
print(data)