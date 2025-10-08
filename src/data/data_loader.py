#load and preprocess offstim data
import os
import h5py
import numpy as np

import scipy.io
from scipy.signal import resample,iirnotch, filtfilt, butter

from utils.data_plotting import plot_psd_comparison, plot_time_comparison



def load_and_normalize_mat(mat_path):
    with open(mat_path, 'rb') as f:
        header = f.read(128)
    
    is_v73 = b'MATLAB 7.3' in header

    if not is_v73:
        # Use scipy.io for version <= 7.2
        mat = scipy.io.loadmat(mat_path)
        data = mat['matrix_rec']
        fs = float(mat['fs'].squeeze())

    else:
        # Use h5py for v7.3 files
        with h5py.File(mat_path, 'r') as f:
            data = f['matrix_rec'][:].T  # Transpose due to MATLAB column-major format
            fs = float(f['fs'][0][0])


    # Normalize each contact (z-score)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data_norm = (data - mean) / (std + 1e-8)

    return data_norm, fs

def downsample(data, original_fs, target_fs):
        num_samples = int(data.shape[1] * target_fs / original_fs)
        return resample(data, num_samples, axis=1)

def segment_data(data, segment_length, fs):
    """
    Args:
        data: (num_contacts, time)
        segment_length: in seconds
        fs: sampling frequency
    Returns:
        list of (num_contacts, segment_samples) arrays
    """
    segment_samples = int(segment_length * fs)
    num_segments = data.shape[1] // segment_samples
    segments = [
        data[:, i*segment_samples : (i+1)*segment_samples]
        for i in range(num_segments)
    ]
    return segments

def load_paired_segments_with_filtering(folder_path, segment_length=1.0, channel_idx=0, cutoff=50, order=8, plot=False):
    """
    loads the data, downsamples, applies notch filter and low pass filter to remove stim artifact 
    (even for offstim data for consistency). then segments the data.

    Args:
        folder_path: local folder path where the mat data is saved for each subject and each hemisphere
        segment_length: in seconds
        cutoff: the low pss frequency
        order: order of LPF
        channel_idx: the example channel to plot for checking the filter
        plot: if we want to plot psd and time series to see the effect of filtering
    Returns:
        segmented data for GPi and STN and new fs
    """
    gpi_data, fs = load_and_normalize_mat(os.path.join(folder_path, 'GPi_Off.mat'))
    stn_data, _ = load_and_normalize_mat(os.path.join(folder_path, 'STN_Off.mat'))

    target_fs = 500
    gpi_data = downsample(gpi_data, fs, target_fs)
    stn_data = downsample(stn_data, fs, target_fs)

    # # Plot PSD before and after filtering for a channel (example: GPi and STN channel 0)
    gpi_filtered = apply_notch_filter(gpi_data, target_fs)
    stn_filtered = apply_notch_filter(stn_data, target_fs)

    # Apply high-order low-pass filter
    gpi_filtered = apply_lowpass_filter(gpi_filtered, target_fs, cutoff, order)
    stn_filtered = apply_lowpass_filter(stn_filtered, target_fs, cutoff, order)

    if plot:
        plot_psd_comparison(gpi_data, gpi_filtered, target_fs, channel_idx)
        plot_psd_comparison(stn_data, stn_filtered, target_fs, channel_idx)

        plot_time_comparison(gpi_data, gpi_filtered, target_fs, channel_idx)
        plot_time_comparison(stn_data, stn_filtered, target_fs, channel_idx)


    gpi_segments = segment_data(gpi_filtered, segment_length, target_fs)
    stn_segments = segment_data(stn_filtered, segment_length, target_fs)

    print("segments of gpi", np.shape(gpi_segments))
    print("segments of stn", np.shape(stn_segments))

    num_pairs = min(len(gpi_segments), len(stn_segments))
    gpi_segments = gpi_segments[:num_pairs]
    stn_segments = stn_segments[:num_pairs]

    return gpi_segments, stn_segments, target_fs

def load_paired_segments_onstim_with_filtering(folder_path,freq, segment_length=1.0, channel_idx=0, cutoff=50, order=8, plot = False):
    gpi_data, fs = load_and_normalize_mat(os.path.join(folder_path, f'GPi_{freq}.mat'))
    stn_data, _ = load_and_normalize_mat(os.path.join(folder_path, f'STN_{freq}.mat'))

    target_fs = 500
    gpi_data = downsample(gpi_data, fs, target_fs)
    stn_data = downsample(stn_data, fs, target_fs)

    # Plot PSD before and after filtering for a channel (example: GPi and STN channel 0)
    gpi_filtered = apply_notch_filter(gpi_data, target_fs)
    stn_filtered = apply_notch_filter(stn_data, target_fs)

    # Apply high-order low-pass filter
    gpi_filtered = apply_lowpass_filter(gpi_filtered, target_fs, cutoff, order)
    stn_filtered = apply_lowpass_filter(stn_filtered, target_fs, cutoff, order)

    if plot:
        plot_psd_comparison(gpi_data, gpi_filtered, target_fs, channel_idx)
        plot_psd_comparison(stn_data, stn_filtered, target_fs, channel_idx)

        plot_time_comparison(gpi_data, gpi_filtered, target_fs, channel_idx)
        plot_time_comparison(stn_data, stn_filtered, target_fs, channel_idx)

    gpi_segments = segment_data(gpi_filtered, segment_length, target_fs)
    stn_segments = segment_data(stn_filtered, segment_length, target_fs)

    print("segments of gpi", np.shape(gpi_segments))
    print("segments of stn", np.shape(stn_segments))

    num_pairs = min(len(gpi_segments), len(stn_segments))
    gpi_segments = gpi_segments[:num_pairs]
    stn_segments = stn_segments[:num_pairs]

    return gpi_segments, stn_segments, target_fs


def build_dataset_with_lag(gpi_segs, stn_segs, lags=3): 
    """
    Creates lagged input and output segments from GPi and STN.

    Args:
        gpi_segs, stn_segs: the segmented data (segments, channels, time)
        lags: number of data points we want to lag

    Returns:
        X: (segments, reg1_channels * (lags+1), time - lags)
        y: (segments, reg2_channels * (lags+1), time - lags)
    """
    # # If we got (N, T, C), swap to (N, C, T)
    # if gpi_segs.shape[1] > gpi_segs.shape[2]:
    #     gpi_segs = np.transpose(gpi_segs, (0, 2, 1))
    #     stn_segs = np.transpose(stn_segs, (0, 2, 1))

    X, y = [], []
    for gpi, stn in zip(gpi_segs, stn_segs):
        # --- GPi ---
        lagged_input = [gpi[:, i:gpi.shape[1] - lags + i] for i in range(lags + 1)]
        X_lag = np.concatenate(lagged_input, axis=0)  # (reg1_channels * (lags+1), time - lags)

        # --- STN (also lagged) ---
        lagged_stn = [stn[:, i:stn.shape[1] - lags + i] for i in range(lags + 1)]
        y_lag = np.concatenate(lagged_stn, axis=0)  # (reg2_channels * (lags+1), time - lags)

        X.append(X_lag)
        y.append(y_lag)

    return np.stack(X), np.stack(y)  # (segments, reg1_channels, time), (segments, reg2_channels, time)

def apply_notch_filter(data, fs, freqs=[60], Q=40):
    #assuming input has shape channels, time
    data_t = np.transpose(data, (1, 0))  # [time, channels]
    filtered = data_t.copy()

    for f0 in freqs:
        b, a = iirnotch(f0, Q, fs)
        for ch in range(filtered.shape[1]):
            filtered[:, ch] = filtfilt(b, a, filtered[:, ch])

    return np.transpose(filtered, (1, 0))  # back to [channels, time]

def apply_lowpass_filter(data, fs, cutoff=50, order=8):
    """
    Apply a lowpass Butterworth filter to multi-channel time series data.
    
    Args:
        data (numpy array): shape (channels, time)
        fs (float): sampling frequency
        cutoff (float): cutoff frequency in Hz
        order (int): filter order

    Returns:
        filtered_data (numpy array): same shape as input
    """
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    filtered = data.copy()
    for ch in range(data.shape[0]):
        filtered[ch, :] = filtfilt(b, a, data[ch, :])
    return filtered

