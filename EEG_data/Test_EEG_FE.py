#================================================
# Final Project
# PSYCH 420
# Wasam Syed (20746474) & Riad Dajani (20816768)
#================================================

# ---Importing necessary libraries
import scipy.io as si
import numpy as np
import matplotlib.pyplot as plt
import random as r
from scipy import fftpack as fftpack
from scipy import signal as sig
import statistics
# import mouse

# ---Setting up variables
# The data_range can be varied from 100 to 38401 (representing 2.5 minutes in the original dataset). 
# channel_number ranges from 1-64 representing all the electrodes according to the original dataset's assignment.
data_range = 38401
time_step = 0.004
channel_number = 25
c_ind = channel_number-1


# (1. SIGNAL ACQUISITION) Link to dataset: https://github.com/mastaneht/SPIS-Resting-State-Dataset/tree/master/Pre-SART%20EEG 
mat_cl_1 = si.loadmat("S02_restingPre_EC.mat")
mat_open_1 = si.loadmat("S02_restingPre_EO.mat")

eyes_closed = mat_cl_1['dataRest'].tolist()
eyes_open = mat_open_1['dataRest'].tolist()
main_data_cl = eyes_closed[c_ind][0:data_range]
main_data_open = eyes_open[c_ind][0:data_range]

ts = [elem for elem in range(data_range)]

plt.figure()
plt.plot(ts, main_data_cl, label = "Closed")
plt.plot(ts, main_data_open, label = "Open")
plt.ylabel("Amp (V)")
plt.xlabel("Time (s)")
plt.suptitle("Raw Signal")
plt.legend()
plt.show()


# Data Formatting
cl_mean = statistics.mean(main_data_cl)    
open_mean = statistics.mean(main_data_open)   

cl_final = [((elem - cl_mean)) for elem in main_data_cl]
open_final = [((elem - open_mean)) for elem in main_data_open]

plt.figure()
plt.plot(ts, cl_final, label = "Closed")
plt.plot(ts, open_final, label = "Open")
plt.ylabel("Amp (V)")
plt.xlabel("Time (s)")
plt.suptitle("Standardized Signal w/ Noise")
plt.legend()
plt.show()


# FFT
def deNoise(signal):
    fourier = fftpack.fft(signal)

    return fourier

denoise_cl = deNoise(cl_final)
denoise_open = deNoise(open_final)

plt.figure()
plt.plot(ts, denoise_cl, label = "Closed")
plt.plot(ts, denoise_open, label = "Open")
plt.suptitle("Denoised Signal")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

# Feature Extraction
def filter_eeg_data(eeg_fft):
    Amp_eeg = np.abs(eeg_fft)

    sample_freq_eeg = fftpack.fftfreq(np.asarray(eeg_fft).size)
    Amp_Freq_eeg = np.array([Amp_eeg, sample_freq_eeg])
    Amp_pos_eeg = Amp_Freq_eeg[0,:].argmax()
    peak_freq_eeg = Amp_Freq_eeg[1, Amp_pos_eeg]

    hf_fft = eeg_fft.copy()
    hf_fft[np.abs(sample_freq_eeg) > peak_freq_eeg] = 0
    filtered_eeg = fftpack.ifft(hf_fft)

    return filtered_eeg

cl_filtered = filter_eeg_data(denoise_cl)
open_filtered = filter_eeg_data(denoise_open)

plt.figure()
plt.plot(ts, cl_filtered, label = "Closed")
plt.plot(ts, open_filtered, label = "Open")
plt.suptitle("Feature Extraction")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

# Spectral Analysis
def compute_psd(signal):
    n = len(signal)
    fft = fftpack.fft(signal, n)
    psd = fft * np.conj(fft)/n
    threshold = 1
    psd_idxs = np.where(psd < threshold, 0, 1)
    freq = (1/(time_step*n)) * np.arange(n)
    psd_clean = psd * psd_idxs

    return psd_clean, freq
psd_closed = compute_psd(denoise_cl)
psd_open = compute_psd(denoise_open)

plt.figure()
plt.plot(psd_closed[1], psd_closed[0], label = "Closed")
plt.plot(psd_open[1], psd_open[0], label = "Open")
plt.suptitle("PSD Analysis")
plt.ylabel("Amplitude (V)")
plt.xlabel("Frequency (Hz)")
plt.legend()
plt.show()

# WIP
# def get_peak_frequency(filtered_eeg):
#     filt_fft_eeg = fftpack.fft(filtered_eeg)

#     Amp_filt_eeg = np.abs(filt_fft_eeg)
#     sample_freq_eeg_filt = fftpack.fftfreq(filt_fft_eeg.size)
#     Amp_Freq_eeg_filt = np.array([Amp_filt_eeg, sample_freq_eeg_filt])
#     Amp_pos_filt_eeg = Amp_Freq_eeg_filt[0,:].argmax()
#     peak_freq_filt_eeg = Amp_Freq_eeg_filt[1, Amp_pos_filt_eeg]

#     return Amp_Freq_eeg_filt

# peak_cl = get_peak_frequency(cl_filtered)
# peak_open = get_peak_frequency(open_filtered)

# plt.figure()
# plt.plot(peak_cl[1], peak_cl[0], label = "Closed")
# plt.plot(peak_open[1], peak_open[0], label = "Open")
# plt.suptitle("Spectral Analysis")
# plt.ylabel("Amplitude of FFT")
# plt.xlabel("Frequency (Hz)")
# plt.legend()
# plt.show()

# Created this specifically for the Power Spectral Density Spectrum
# def occipital_data(d, r=data_range):
#   ans_dict = {
#       'O1': d[26][0:r],
#       'O2': d[63][0:r],
#       'Oz': d[28][0:r],
#       'PO7':d[24][0:r],
#       'PO3':d[25][0:r],
#       'POz':d[29][0:r],
#       'PO8':d[61][0:r],
#       'PO4':d[62][0:r]
#   }
#   return ans_dict

# moc_02 = occipital_data(d = eyes_closed)
# mop_02 = occipital_data(d = eyes_open)

# # Feature extraction
# Fs = 256
# freq_cl_02, wel_eeg_cl_02 = sig.welch(moc_02['O1'], Fs, scaling='spectrum')
# freq_open_02, wel_eeg_open_02 = sig.welch(mop_02['O1'], Fs, scaling='spectrum')

# plt.semilogy(freq_cl_02[0:40], wel_eeg_cl_02[0:40], label = 'Closed')
# plt.semilogy(freq_open_02[0:40], wel_eeg_open_02[0:40], label='Open')
# plt.xlabel('frequency (Hz)')
# plt.ylabel('PSD')
# plt.suptitle('Power Spectral Density Spectrum')
# plt.legend()
# plt.grid()
# plt.show()

# # Classification using neural network
# # Backprop Proof of concept (data from eeg goes in, and can identify state by useing mean as classifier)

eeg_data = np.array(cl_final) # convert to numpy array
eeg_data = eeg_data/np.amax(eeg_data, axis=0) # normalize the data

X = np.vstack((cl_filtered.T, open_filtered.T))
Y = np.array(([1], [1], [0], [0], [1], [1], [0], [0]), dtype=float)

class NN(object):
	def __init__(self):
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenLayer = 6

		self.W1 = np.random.randn(self.inputSize, self.hiddenLayer)
		self.W2 = np.random.randn(self.hiddenLayer, self. outputSize)

	def cost(self, X, Y):
		self.yH = self.fforward(X)
		return sum((Y-self.yH)**2)*1/2

	def costP(self, X, Y):
		self.yH = self.fforward(X)
		d3 = np.multiply(-(Y-self.yH), self.sigP(self.z3))
		costW2 = np.dot(self.a2.T, d3)
		d2 = np.dot(d3, self.W2.T)*self.sigP(self.z2)
		costW1 = np.dot(X.T, d2)
		return costW1, costW2

	def sig(self, z):
		return 1/(1+np.exp(-z))

	def sigP(self, z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def fforward(self, x):
		self.z2 = np.dot(x, self.W1)
		self.a2 = self.sig(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yH = self.sig(self.z3)
		return yH

NN = NN()
error = []
iteration = []
for i in range(1000):
	J1 = NN.cost(X, Y)
	t = 3
	costW1, costW2 = NN.costP(X, Y)
	NN.W1 = NN.W1 - t*costW1
	NN.W2 = NN.W2 - t*costW2
	J2 = NN.cost(X,Y)
	costW1, costW2 = NN.costP(X, Y)
	NN.W1 = NN.W1 - t*costW1
	NN.W2 = NN.W2 - t*costW2
	J3 = NN.cost(X,Y) 

	yH = NN.fforward(X)
	error.append(J3)
	iteration.append(i)
	if (i == 1000):
		break

print(np.round(yH, 2))
plt.plot(iteration, error)
plt.show()

# # APPLICATION

# # The following snippet of code moves the mouse up if the output of the 
# # perceptron is True i.e., the eyes are open.

# # Commented out to prevent program from crashing

# """
# def move_mouse(op):
#   if (inp == 1):
#     mouse.move(0, -1000, absolute=False, duration=0.01)
#   else:
#     mouse.move(0, 1000, absolute=False, duration=0.01)
# """