import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks


# === Helper functions ===

def autocorr(x):
    """Compute normalized autocorrelation."""
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    return result / np.max(result)


def detect_voiced_region(signal, fs, frame_size=0.02, threshold=0.1):
    """
    Detects a voiced region (high-energy region) in the signal.
    Returns start index of the first voiced frame.
    """
    frame_len = int(frame_size * fs)
    energy = [
        np.sum(signal[i:i + frame_len] ** 2)
        for i in range(0, len(signal) - frame_len, frame_len)
    ]
    energy = np.array(energy) / np.max(energy)

    # Find first region with energy above threshold
    voiced_indices = np.where(energy > threshold)[0]
    if len(voiced_indices) == 0:
        return len(signal) // 2  # fallback to middle if nothing found
    start_frame = voiced_indices[0]
    return start_frame * frame_len


def analyze_speaker(file_path):
    # === Step 1: Read the audio file ===
    fs, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # convert to mono
    data = data.astype(np.float32)

    # === Step 2: Detect vowel (voiced) region automatically ===
    start_index = detect_voiced_region(data, fs)
    duration = 0.03  # 30 ms
    num_samples = int(fs * duration)
    vowel_part = data[start_index:start_index + num_samples]

    # === Step 3: Compute autocorrelation ===
    ac = autocorr(vowel_part)

    # === Step 4: Find peaks ===
    peaks, _ = find_peaks(ac, height=0)
    if len(peaks) < 2:
        return None, None, ac  # if no peaks detected

    # Ignore lag=0; take first peak after that
    lag_samples = peaks[0]
    lag_time = lag_samples / fs
    f0 = 1 / lag_time
    return f0, fs, ac


# === Analyze both speakers ===
male_f0, fs_male, ac_male = analyze_speaker("MaskMale.wav")
female_f0, fs_female, ac_female = analyze_speaker("MaskFemale.wav")

# === Plot autocorrelation ===
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(ac_male)
plt.title(f"Male Speaker Autocorrelation — F₀ ≈ {male_f0:.2f} Hz")
plt.xlabel("Lag (samples)")
plt.ylabel("Autocorrelation")

plt.subplot(2, 1, 2)
plt.plot(ac_female)
plt.title(f"Female Speaker Autocorrelation — F₀ ≈ {female_f0:.2f} Hz")
plt.xlabel("Lag (samples)")
plt.ylabel("Autocorrelation")

plt.tight_layout()
plt.show()

print(f"Male Fundamental Frequency ≈ {male_f0:.2f} Hz")
print(f"Female Fundamental Frequency ≈ {female_f0:.2f} Hz")
