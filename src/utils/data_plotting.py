import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

def plot_psd_comparison(before, after, fs, channel_idx=0):
    x_before = before[channel_idx,:]
    x_after = after[channel_idx,:]

    freqs_before, psd_before = welch(x_before, fs=fs, nperseg=fs)
    freqs_after, psd_after = welch(x_after, fs=fs, nperseg=fs)

    plt.figure(figsize=(8, 4))
    plt.semilogy(freqs_before, psd_before, label='Before filter')
    plt.semilogy(freqs_after, psd_after, label='After filter')
    plt.title(f"PSD – Channel {channel_idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_time_comparison(before, after, fs, channel_idx=0, seconds=2):
    """
    Plot time-domain signals before and after filtering.
    
    Args:
        before (np.array): shape [channels, time]
        after (np.array): same shape
        fs (int): sampling rate
        channel_idx (int): which channel to visualize
        seconds (float): how many seconds to display
    """
    samples = int(seconds * fs)
    time = np.arange(samples) / fs

    x_before = before[channel_idx, :samples]
    x_after = after[channel_idx, :samples]

    plt.figure(figsize=(8, 4))
    plt.plot(time, x_before, label='Before filter', alpha=0.7)
    plt.plot(time, x_after, label='After filter', alpha=0.7)
    plt.title(f"Time-Domain Signal – Channel {channel_idx} (First {seconds}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()