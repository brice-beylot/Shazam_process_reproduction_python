import argparse
import librosa
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import butter, filtfilt
import numpy as np
import soundfile as sf


def bandpass_filter(y, sr, lowcut=300, highcut=3700):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, y)


def analyze_music(music_path):
    # Load audio file
    filename = f"Musiques/{music_path}"
    # Resample to 8 kHz for better performance
    y, sr = librosa.load(filename, sr=8000)
    # Apply bandpass filter
    lowcut = 300
    highcut = 3700
    y_filtered = bandpass_filter(y, sr, lowcut, highcut)
    print(
        f"Analyzing: {music_path} (Sample rate: {sr} Hz, Duration: {librosa.get_duration(y=y_filtered, sr=sr):.2f}s)")
    # sf.write("filtered_song.wav", y_filtered, sr)

    # Example: Plot the waveform
    """
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y_filtered, sr=sr)
    plt.title("Waveform")
    plt.show()
    """

    hop_length = 256
    n_fft = 1024
    # Short-Time Fourier Transform
    D = librosa.stft(y_filtered, hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    """
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (300–3700 Hz)')
    plt.show()
    """
    # width of the filter in order to cover half a second
    delta_time = hop_length/sr
    delta_freq = sr/n_fft
    # Creation of a filter that covers 0.5 seconds in the time domain
    time_window = int(0.5 * 1/delta_time)
    # Creation of a filter that covers 200 Hz in the frequency domain
    freq_window = int(200 * 1/delta_freq)
    print(
        f"Filter width (time): {time_window} frames, Filter height (frequency): {freq_window} bins")
    window_shape = (freq_window, time_window)
    # Apply maximum filter
    neighborhood = np.ones(window_shape)
    local_maxima = (S_db == maximum_filter(S_db, footprint=neighborhood))
    # Get coordinates of local maxima (time, frequency)
    peaks = np.argwhere(local_maxima)
    """
    # For each time frame, keep only the top 2 peaks
    filtered_peaks = []
    for t in np.unique(peaks[:, 1]):
        frame_peaks = peaks[peaks[:, 1] == t]
        if len(frame_peaks) > 2:
            # Sort by amplitude and keep top 2
            amplitudes = S_db[frame_peaks[:, 0], frame_peaks[:, 1]]
            top_indices = np.argsort(amplitudes)[-2:]
            filtered_peaks += peaks[np.isin(peaks, frame_peaks[top_indices]
                                            ).all(axis=1)].tolist()
    """
    top_percentile = 96  # Keep top 5% of peaks by amplitude
    threshold_db = np.percentile(S_db, top_percentile)
    filtered_peaks = peaks[S_db[peaks[:, 0], peaks[:, 1]] > threshold_db]

    constellation_map = []
    for peak in filtered_peaks:
        time_sec = peak[1] * (hop_length / sr)
        freq_hz = peak[0] * (sr / n_fft)
        amplitude_db = S_db[peak[0], peak[1]]
        constellation_map.append([time_sec, freq_hz, amplitude_db])
    constellation_map = np.array(constellation_map)
    print(constellation_map)
    print(
        f"S_db shape : {S_db.shape}, Constellation map size: {len(constellation_map)}")

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.scatter(
        constellation_map[:, 0],
        constellation_map[:, 1],
        c='red', s=10, marker='x'
    )
    plt.title('Constellation Map (Corrected)')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


if __name__ == "__main__":
    print("Welcome to the Shazam-like Music Analyzer!")
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Shazam-like Music Analyzer")
    parser.add_argument("--music", type=str, required=True,
                        help="Name of the music file (MP3/WAV)")
    args = parser.parse_args()

    # Analyze the music
    analyze_music(args.music)
