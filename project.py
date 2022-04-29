import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.signal import stft, istft
import sounddevice as sd


def inv_STFT(X, fs, winlen, window):
    """
    Inverse STFT function

    :param X: time-varrying input signal
    :param fs: Sampling frequency
    :param winlen: Window length
    :param window: Desired window to use
    """
    _, x = istft(X, fs=fs, window=window, nperseg=winlen,
                 noverlap=winlen / 2, nfft=winlen, input_onesided=True)
    return x


def update(x, fs, H_updated, P_updated, k, alpha):
    """
    Update the harmonic and precussive components iteratively and binarize the seperation result

    :param x: time-varrying input signal
    :param fs: Sampling frequency
    :param H_updated: Harmonic component
    :param P_updated: Percussive component
    :param k: Value of k
    :param alpha: Alpha value

    :returns [H_max,P_max]:
    """
    [h_len, i_len] = np.shape(H_updated[0])
    for k_index in np.arange(k):
        H0 = H_updated[k_index]
        P0 = P_updated[k_index]
        H1 = np.zeros(np.shape(H0))
        P1 = np.zeros(np.shape(P0))

        # update harmonic and percussive components by calculating gradient value
        for h in np.arange(0, h_len-1):
            for i in np.arange(0, i_len-1):
                max = float(
                    np.max([H0[h, i]+gradient(H0, P0, h, i, alpha), 0]))
                w = float(W[h, i])
                H1[h, i] = np.minimum(max, w)
                P1[h, i] = W[h, i]-H1[h, i]
        H_updated.append(H1)
        P_updated.append(P1)

        # Plot where k is 0, 5, 10 and 50
        if k_index in [0, 5, 10, 50]:
            print(k_index)
            plot_updated_components(
                x, fs, H_updated[-1], P_updated[-1], k_index)

    # Take the last updated elements
    H_max_1 = H_updated[-1]
    P_max_1 = P_updated[-1]
    H_max = np.zeros(np.shape(H_max_1))
    P_max = np.zeros(np.shape(P_max_1))

    # Binarize the separation result
    for i in np.arange(h_len):
        for j in np.arange(i_len):
            if H_max_1[i, j] < P_max_1[i, j]:
                H_max[i, j] = 0
                P_max[i, j] = W[i, j]
            else:
                H_max[i, j] = W[i, j]
                P_max[i, j] = 0

    return [H_max, P_max]


def gradient(H0, P0, h, i, alpha):
    """
    Calculate the gradient of corresponding parameters

    :param H0:
    :param P0: 
    :param h:
    :param i:
    :param alpha:
    :returns grad: 
    """
    partH = H0[h, i-1]-2 * H0[h, i] + H0[h, i+1]
    partP = P0[h-1, i]-2 * P0[h, i] + P0[h+1, i]
    grad = alpha * partH/4 - (1-alpha) * partP/4
    return grad


def convert(H_max, P_max, gamma):
    """
    Convert HMAX and PMAX into waveforms

    :param H_max:
    :param P_max: 
    :param gamma:
    :returns [h,p]: 
    """
    H_M = (H_max ** (1 / (2 * gamma))) * np.exp(1j*np.angle(F))
    P_M = (P_max ** (1 / (2 * gamma))) * np.exp(1j*np.angle(F))
    h = inv_STFT(H_M, fs, winlen, han)
    p = inv_STFT(P_M, fs, winlen, han)
    return [h, p]


def plot_updated_components(x, fs, H_updated, P_updated, k):
    """
    Plots the harmonic and percussive spectrums

    :param x: time-varrying input signal
    :param fs: Sampling frequency
    :param H_updated:
    :param P_updated:
    :param k: Value of k
    """
    # Plot the harmonic spectrum
    [n, m] = np.shape(H_updated)
    fv = np.linspace(0, fs / 2, n)
    tv = np.linspace(0, int(float(len(x)) / float(fs)), m)
    plt.subplot(1, 2, 1)
    plt.pcolormesh(tv, fv, H_updated, shading='auto')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Harmonic')

    # Plot the percussive spectrum
    plt.subplot(1, 2, 2)
    plt.pcolormesh(tv, fv, P_updated, shading='auto')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Percussive')
    plt.suptitle('Saperation result at k=' + str(k))
    plt.show()
    [h, p] = convert(H_updated, P_updated, gamma)
    print("The SNR with k="+str(k)+" is "+str(evaluate(x, h, p)))


def plot_results(x, W, fs, H_max, P_max, k):
    # Plot the harmonic spectrogram
    [n, m] = np.shape(H_max)
    fv = np.linspace(0, fs / 2, n)
    tv = np.linspace(0, int(float(len(x)) / float(fs)), m)

    plt.subplot(3, 1, 1)
    plt.pcolormesh(tv, fv, H_max, shading='auto')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Harmonic Component Spectrogram')
    plt.tight_layout(pad=0.1)

    # Plot the percussive spectrogram
    plt.subplot(3, 1, 2)
    plt.pcolormesh(tv, fv, P_max, shading='auto')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Percussive Component Spectrogram')
    plt.tight_layout(pad=0.1)

    # Plot power spectrogram of the original signal
    plt.subplot(3, 1, 3)
    [n, m] = np.shape(W)
    fv = np.linspace(0, fs / 2, n)
    tv = np.linspace(0, int(float(len(x)) / float(fs)), m)
    plt.pcolormesh(tv, fv, W, shading='auto')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Original signal spectrogram')
    name = 'seperation_results_k_' + str(k)
    plt.savefig(name)
    plt.show()


def play_sounds(fs, h, p, k):
    """
    Saves and play the seperated audio materials

    :param fs: Sampling frequency
    :param h: Harmonic component
    :param p: Percussive component
    :param k: Value of k
    """
    n = 1000
    print("Playing Harmonic Component")
    name = 'harmonic_sqrthan_' + str(k) + '.wav'
    y = np.asarray(h/np.abs(np.max(h)) * n, dtype=np.int16)
    wavfile.write(name, fs, np.asarray(
        h/np.abs(np.max(h)) * n, dtype=np.int16))
    sd.play(y, fs)
    sd.wait()

    print("Now Playing Percussive Component")
    name = 'percussive_sqrthan_' + str(k)+'.wav'
    y = np.asarray(p/np.abs(np.max(p)) * n, dtype=np.int16)
    wavfile.write(name, fs, np.asarray(
        p/np.abs(np.max(p)) * n, dtype=np.int16))
    sd.play(y, fs)
    sd.wait()


def evaluate(x, h, p):
    """
    Calculate the SNR

    :param x: Time series of measurement values
    :param h: Harmonic component
    :param p: Percussive component 
    :return SNR: SNR 
    """
    x = x.astype('float64')
    m = (h + p).astype('float64')
    e = x-m[:len(x)]
    x_power = np.sum(np.power(x, 2))
    e_power = np.sum(np.power(e, 2))
    SNR = 10 * np.log10(x_power / e_power)
    return SNR


if __name__ == "__main__":
    # Load the test material
    fs, x = wavfile.read('project_test1.wav')
    x = np.asarray(x).astype('float64')

    # Intialize the necessary parameters
    k = 51
    gamma = 0.3
    alpha = 0.3
    winlen_ms = 20.0
    winlen = int(2.0 ** np.ceil(np.log2(float(fs) * winlen_ms / 1000.0)))
    han = signal.get_window('hann', int(winlen))

    # Calculate STFT of the input signal
    _, _, F = stft(x, fs, window=han, nperseg=winlen, noverlap=winlen / 2, nfft=winlen, detrend=False,
                return_onesided=True, padded=True, axis=-1)

    # Set initial values
    W = (np.abs(F)) ** (2*gamma)
    H = P = W / 2
    H_updated = [H]
    P_updated = [P]

    # Calculate the update variables
    [H_max, P_max] = update(x, fs, H_updated, P_updated, k, alpha)
    [h, p] = convert(H_max, P_max, gamma)
    print("The SNR in this test is "+str(evaluate(x, h, p)))

    # Evaluate through images and seperated audio materials.
    #play_sounds(fs, h, p, k)
    plot_results(x, W, fs, H_max, P_max, k)
