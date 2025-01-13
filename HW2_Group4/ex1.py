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
import argparse
import uuid
from dataclasses import dataclass
from redis import Redis


class HTReader:
    """An HTReader is used to read humidity and temperature from the sensor."""

    def __init__(self) -> None:
        self.dht = adafruit_dht.DHT11(D4)
        self.mac_adress = hex(uuid.getnode())


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


@dataclass
class RedisCredential:
    """
    The RedisCredential is a class used to store users data.
    """
    username: str
    password: str
    host: str
    port: str

TanguyRedis = RedisCredential(
    "default",
    "2llMg9E9AkqLcDdx6HDpLQ7nzkQKgtCC",
    "redis-17061.c300.eu-central-1-1.ec2.redns.redis-cloud.com",
    17061
)

def parse_credentials():
    """Parse the credentials from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--user", default='default')
    parser.add_argument("--password", default=None)
    args = parser.parse_args()
    if args.host is None:
        raise ValueError("Host can't be None, please specify an host with --host")
    if args.port is None:
        raise ValueError("Port can't be None, please specify a port with --port")
    if args.user is None:
        print("No user was mentioned, default value used is 'default'. You can change it by specifying --user")
    if args.password is None:
        raise ValueError("Password can't be None, please specify a password with --password")
    return RedisCredential(args.user, args.password, args.host, args.port)

class RedisWriter:
    """
    The RedisWriter class is used to manage the redis database.
    """

    def __init__(
        self,
        credentials: RedisCredential,
        timeseries: list[str],
        chunk_size_bytes: int = 4096,
        retention_hour: int = 24,
        compresses: bool = True,
        delete_if_exists: bool = False
    ):
        # Create the redis client
        self.redis_client = Redis(
            host=credentials.host,
            port=credentials.port,
            username=credentials.username,
            password=credentials.password
        )
        
        # Delete the previous measures if needed.
        if delete_if_exists:
            for ts in timeseries:
                try:
                    self.redis_client.delete(ts)
                except:
                    pass

        # Create the timeseries
        for ts in timeseries:
            try:
                self.redis_client.ts().create(
                    ts,
                    retention_msecs=retention_hour*60*60*1000,
                    chunk_size=chunk_size_bytes,
                    uncompressed = not compresses
                )
            except:
                pass
        
        # Verify the connection
        if self.redis_client.ping():
            print("Succesfully connected to Redis")
        else:
            print("Not connected to Redis.")
    
    def add(self, ts, value):
        """Add a new value to the time series."""
        timestamp = int(time()*1000)
        self.redis_client.ts().add(ts, timestamp, value)

# Fix the CPU frequency to its maximum value (1.5 GHz)
Popen('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"', shell=True).wait()

# Parameters for audio recording and VAD
SAMPLE_RATE = 48000  # Sampling rate for recording in Hz.
DOWNSAMPLED_RATE = 16000  # Higher downsample rate for faster processing.
CHANNELS = 1 # The number of channels of the audio recording.
DHT_PIN = 4  # GPIO pin where the DHT sensor is connected.

FRAME_LENGTH = 0.064 # [s] The length in seconds of the frame.
FRAME_STEP = 0.032 # [s] The difference in secondes between two frames.
SILENCE_THRESHOLD = 15  # The threshold below which recorded audio is considered as slience.
DURATION_THRESHOLD = 0.3  # Minimum duration of non-silence to consider as voice.

NUM_MEL_BINS = 100 # The number of binsfor the Mel Spectrogram processing
LOWER_FREQUENCY = 20 # [Hz] The lower frequency for the Mel Spectrogram
UPPER_FREQUENCY = 1000 # [Hz] The upper frequency for the mel spectrogram
NUM_COEFFICIENTS = 10 # The number of coefficient kept for the MFCC

class Normalization():
    def __init__(self, bit_depth):
        self.max_range = bit_depth.max

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

class MelSpectrogram():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
    ):
        self.spec_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spec_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency,
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spec_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

    def get_mel_spec_and_label(self, audio, label):
        log_mel_spectrogram = self.get_mel_spec(audio)

        return log_mel_spectrogram, label

class MFCC():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency,
        num_coefficients
    ):

        self.mel_spec_processor = MelSpectrogram(
            sampling_rate, frame_length_in_s, frame_step_in_s, num_mel_bins, lower_frequency, upper_frequency
        )
        self.num_coefficients = num_coefficients

    def get_mfccs(self, audio):
        log_mel_spectrogram = self.mel_spec_processor.get_mel_spec(audio)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def get_mfccs_and_label(self, audio, label):
        mfccs = self.get_mfccs(audio)

        return mfccs, label

class VAD():
    def __init__(
        self,
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        dBthres,
        duration_thres,
    ):
        self.frame_length_in_s = frame_length_in_s
        self.frame_step_in_s = frame_step_in_s
        self.spec_processor = Spectrogram(
            sampling_rate, frame_length_in_s, frame_step_in_s,
        )
        self.dBthres = dBthres
        self.duration_thres = duration_thres

    def is_silence(self, audio):
        spectrogram = self.spec_processor.get_spectrogram(audio)

        dB = 20 * tf.math.log(spectrogram + 1.e-6)
        energy = tf.math.reduce_mean(dB, axis=1)
        min_energy = tf.reduce_min(energy)

        rel_energy = energy - min_energy
        non_silence = rel_energy > self.dBthres
        non_silence_frames = tf.math.reduce_sum(tf.cast(non_silence, tf.float32))
        non_silence_duration = self.frame_length_in_s + self.frame_step_in_s * (non_silence_frames - 1)

        if non_silence_duration > self.duration_thres:
            return 0
        else:
            return 1

# Initialize instances
normalization = Normalization(tf.int16)
vad_processor = VAD(DOWNSAMPLED_RATE, FRAME_LENGTH, FRAME_STEP, SILENCE_THRESHOLD, DURATION_THRESHOLD)
# We are aware that by using the default VAD and MFCC, the call computation of the spectrogram of the same audio tensor will be done twice
# Which is a waste of resources (once for the VAD and once for the KWS).
processor = MFCC(DOWNSAMPLED_RATE, FRAME_LENGTH, FRAME_STEP, NUM_MEL_BINS, LOWER_FREQUENCY, UPPER_FREQUENCY, NUM_COEFFICIENTS)

class KeyWordSpotter:

    def __init__(self, model_path):

        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details() 
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, input_tensor):
        input_tensor = tf.reshape(input_tensor, tuple(self.input_details[0]['shape']))

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]


model_path = "model4.tflite"
model = KeyWordSpotter(model_path)

# initliaze the data
data_collection_enabled = False

def resample(audio_data, sample_rate, downsample_rate):
    return sps.resample_poly(audio_data, up=1, down=sample_rate/downsample_rate)

LABELS = ['down', 'up']

def audio_callback(indata, frames, tim, status):
    """Continuously record and analyze audio to toggle data collection state."""
    global data_collection_enabled

    audio_data = np.float32(indata[:, 0])
    
    downsampled_audio = resample(audio_data, SAMPLE_RATE, DOWNSAMPLED_RATE)

    # Convert to TensorFlow tensor, squeeze, and normalize
    normalized_audio = normalization.normalize_audio(downsampled_audio)
    squeezed_audio = tf.squeeze(normalized_audio)
    # Check if the audio contains speech using the keyword
    if not vad_processor.is_silence(squeezed_audio):
        # If yes, check if the word is the one we are looking for.
        target = LABELS.index('down') if data_collection_enabled else LABELS.index('up')
        input_data = processor.get_mfccs(normalized_audio)
        output = model(input_data)

        if output[target] >= 0.99:
            # Toggle data collection state
            data_collection_enabled = not data_collection_enabled
            if data_collection_enabled:
                print(f"Data collection enabled, say 'down' to disable it.")
            else:
                print(f"Data collection disabled, say 'up' to enable it.")

def main(parse_creds):
    """Main algorithm."""

    # Initiliaze the sensor.
    htreader = HTReader()

    # Connect to redis and create time series.
    if parse_creds:
        credentials = parse_credentials()
    else:
        credentials = TanguyRedis
    tempts = 'temperature:' + htreader.mac_adress
    humts = 'humidity:' + htreader.mac_adress
    rdwriter = RedisWriter(credentials, (tempts, humts))

    # Start recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=audio_callback, blocksize=SAMPLE_RATE):
        print("VAD is running. Say 'up' to toggle data collection. Currently disabled.")
        while True:
            t0 = time()
            if data_collection_enabled: # read the temperature and the humidity
                humidity, temperature = htreader.humidity(), htreader.temp()
                if humidity is not None: # send them to redis
                    rdwriter.add(humts, humidity)
                if temperature is not None:
                    rdwriter.add(tempts, temperature)
            dt = time() - t0
            sleep_pause = 2 - dt
            if sleep_pause > 0:
                sleep(sleep_pause)  # Wait for the remaining time of the 2 seconds to compute again.

if __name__ == "__main__":
    parse_creds = True
    main(parse_creds)
