"""
This file contains the second exercise of the homework 1 of the course
'Machine Learning for IoT'
2024 - 2025
Group 4
"""
import tensorflow as tf
import numpy as np
import sounddevice as sd
import adafruit_dht
from time import time, sleep
from subprocess import Popen
from board import D4
import scipy.signal as sps

class HTReader:
    """An HTReader is used to read humidity and temperature from the sensor."""

    def __init__(self) -> None:
        self.dht = adafruit_dht.DHT11(D4)

    def humidity(self):
        """Read the humidity."""
        try:
            humidity = self.dht.humidity
        except:
            humidity = None
            self.dht.exit()
            self.dht = adafruit_dht.DHT11(D4)

        return humidity

    def temp(self):
        """Read the temperature."""
        try:
            temp = self.dht.temperature
        except:
            temp = None
            self.dht.exit()
            self.dht = adafruit_dht.DHT11(D4)
        return temp

# Fix the CPU frequency to its maximum value (1.5 GHz)
Popen('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"', shell=True).wait()

# Parameters for audio recording and VAD
SAMPLE_RATE = 48000  # Sampling rate for recording in Hz
DOWNSAMPLED_RATE = 16000  # Higher downsample rate for faster processing
CHANNELS = 1 # The number of channels of the audio recording
SILENCE_THRESHOLD = 15  # Lower threshold for improved sensitivity
DURATION_THRESHOLD = 0.3  # Minimum duration of non-silence to consider as voice
TOGGLE_COOLDOWN = 5  # Minimum time (in seconds) between state toggles

# Initialize the VAD class
class Normalization():
    def __init__(self, bit_depth):
        self.max_range = np.iinfo(bit_depth).max

    def normalize_audio(self, audio):
        audio_float32 = tf.cast(audio, tf.float32)
        audio_normalized = audio_float32 / self.max_range
        return audio_normalized

    def normalize(self, audio, label):
        audio_normalized = self.normalize_audio(audio)
        return audio_normalized, label

class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_spectrogram_and_label(self, audio, label):
        spectrogram = self.get_spectrogram(audio)

        return spectrogram, label

class VAD():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s, dBthres, duration_thres):
        self.spec_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        self.dBthres = dBthres
        self.duration_thres = duration_thres
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s

    def is_silence(self, audio):
        """Return True if the audio array is silent or not."""
        spectrogram = self.spec_processor.get_spectrogram(audio)
        dB = 20 * tf.math.log(spectrogram + 1.e-6)
        # Compute the energy of the spectrogram
        energy = tf.math.reduce_mean(dB, axis=1)
        min_energy = tf.reduce_min(energy)

        rel_energy = energy - min_energy
        #  Verify which frame is silence
        non_silence = rel_energy > self.dBthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        # Compute the silent duration from the number of silent frame
        non_silence_duration = self.frame_length_in_s + self.frame_step_in_s * (non_silence_frames - 1)

        return non_silence_duration < self.duration_thres

# Initialize instances
normalization = Normalization(np.int16)
vad_processor = VAD(DOWNSAMPLED_RATE, 0.03, 0.015, SILENCE_THRESHOLD, DURATION_THRESHOLD)
data_collection_enabled = True
last_toggle_time = 0

def resample(audio_data, sample_rate, downsample_rate):
    sps.resample_poly(audio_data, up=1, down=sample_rate/downsample_rate)

def audio_callback(indata, frames, tim, status):
    """Continuously record and analyze audio to toggle data collection state. """
    global data_collection_enabled, last_toggle_time

    # Convert audio data to np.float32 and downsample to 16 kHz
    audio_data = np.float32(indata[:, 0])
    
    # Downsample the audio data
    downsampled_audio = resample(audio_data, SAMPLE_RATE, DOWNSAMPLED_RATE)

    # Convert to TensorFlow tensor, squeeze, and normalize
    audio_tensor = tf.convert_to_tensor(downsampled_audio, dtype=tf.float32)
    audio_tensor = tf.squeeze(audio_tensor)
    audio_normalized = normalization.normalize_audio(audio_tensor)

    # Check if the audio contains speech using VAD
    if not vad_processor.is_silence(audio_normalized) and (time() - last_toggle_time > TOGGLE_COOLDOWN):
        # Toggle data collection state
        data_collection_enabled = not data_collection_enabled
        last_toggle_time = time()
        print(f"Data collection {'enabled' if data_collection_enabled else 'disabled'}")

def main():
    """Main algorithm."""
    htreader = HTReader()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=audio_callback, blocksize=SAMPLE_RATE):
        print("VAD is running. Say something to toggle data collection.")
        while True:
            if data_collection_enabled:
                humidity, temperature = htreader.humidity(), htreader.temp()
                # Print the data collected
                if humidity is not None and temperature is not None:
                    print(f"Temperature: {temperature}°C, Humidity: {humidity}%")
                else:
                    print("Failed to retrieve data from the sensor.")
            sleep(2)  # Measure every 2 seconds when enabled


if __name__ == "__main__":
    main()
