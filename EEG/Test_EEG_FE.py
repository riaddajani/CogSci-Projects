import statistics
import numpy as np
import matplotlib.pyplot as mp
from math import exp as exp
import pywt
import scipy.io as si
from scipy import fftpack as fftpack

#Variables
data_range = 1000
time_step = 0.001
channel_number = 25
c_ind = channel_number-1


# Real EEG data

mat_cl_1 = si.loadmat("EEG\S02_restingPre_EC.mat")
mat_open_1 = si.loadmat("EEG\S02_restingPre_EO.mat")

ys_cl = mat_cl_1['dataRest'].tolist()
ys_open = mat_open_1['dataRest'].tolist()

main_data_cl = ys_cl[c_ind][100:data_range]
main_data_open = ys_open[c_ind][100:data_range]

ts = [elem for elem in range(len(main_data_open))]

cl_mean = statistics.mean(main_data_cl) 
open_mean = statistics.mean(main_data_open)

cl_final = [((elem - cl_mean)) for elem in main_data_cl]
open_final = [((elem - open_mean)) for elem in main_data_open]

mp.figure()
mp.plot(ts, cl_final, label = "Closed Signal")
mp.plot(ts, open_final, label = "Open Signal")
mp.ylabel("Amp (V)")
mp.xlabel("Time (s)")
mp.suptitle("Standardized Signal")
mp.legend()
mp.show()

# Decomposition
def decompose_signal(signal_input):
	'''Decompose the signal into its frequency components'''
	coeffs = pywt.wavedec(signal_input, 'db4', level=6)
	return coeffs

cl_signal_coeffs = decompose_signal(cl_final)
open_signal_coeffs = decompose_signal(open_final)

ts = [elem for elem in range(len(cl_signal_coeffs[1]))]

mp.figure()
mp.plot(ts, cl_signal_coeffs[1])
mp.plot(ts, open_signal_coeffs[1])
mp.xlabel('Time (s)')
mp.ylabel("Amp (V)")
mp.title('Decomp of Signal')
mp.show()

# Fourier transform of signal
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

cl_fft = filter_eeg_data(cl_signal_coeffs[1])
open_fft = filter_eeg_data(open_signal_coeffs[1])

print(open_fft)

mp.figure()
mp.plot(ts, cl_fft, label = "Closed Signal")
mp.plot(ts, open_fft, label = "Open Signal")
mp.xlabel('Time (s)')
mp.ylabel("Amp (V)")
mp.title('Feature Extraction of Signal')
mp.show()


# Fourier transform and PSD of signal
def fft(signal, threshold=5):
	'''Calculate the Fourier transform of a signal'''
	n = len(signal)
	fourier = np.fft.fft(signal, n)
	psd = fourier * np.conj(fourier) / n
	freq = (1/(time_step*n)) * np.arange(n)
	psd_idxs = psd > threshold
	psd_clean = psd * psd_idxs
	fourier_clean = fourier * psd_idxs
	filtered_signal = np.fft.ifft(fourier_clean)

	return freq, filtered_signal, psd_clean

mp.figure()
mp.plot(ts, fft(cl_fft)[1])
mp.plot(ts, fft(open_fft)[1])
mp.xlabel('Time (s)')
mp.ylabel("Amp (V)")
mp.title('filtered signal')
mp.show()

mp.figure()
mp.plot(fft(cl_fft)[0], fft(cl_fft)[2])
mp.plot(fft(open_fft)[0], fft(open_fft)[2])
mp.xlabel('Freq (Hz)')
mp.ylabel("Amp (V)")
mp.title('PSD of signal')
mp.show()
