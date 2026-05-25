import argparse
import librosa
import matplotlib.pyplot as plt
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
    y, sr = librosa.load(filename)
    # Apply bandpass filter
    y_filtered = bandpass_filter(y, sr)
    print(
        f"Analyzing: {music_path} (Sample rate: {sr} Hz, Duration: {librosa.get_duration(y=y_filtered, sr=sr):.2f}s)")
    # sf.write("filtered_song.wav", y_filtered, sr)

    # Example: Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y_filtered, sr=sr)
    plt.title("Waveform")
    plt.show()

    D = librosa.stft(y_filtered)  # Short-Time Fourier Transform
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (300–3700 Hz)')
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
