import pyaudio
import numpy as np
from scipy import signal
import librosa

# Parameters
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate in Hz
CHUNK_SIZE = 2048  # Number of frames per buffer
SILENCE_THRESHOLD = 15  #Silence if sound lower than
PITCH_SHIFT_FACTOR = 1.1  # Adjust as needed, 1.0 is no change in pitch
NOISE_SCALE = 0.05
DISTORTION = 0.005

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open an input stream (microphone)
input_stream = p.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK_SIZE)

# Open an output stream (speakers)
output_stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       output=True,
                       frames_per_buffer=CHUNK_SIZE)

# Filter coefficients for a low-pass filter
b, a = signal.butter(5, 4000, 'low', fs=RATE)

try:
    print("Simulating old radio audio... (Press Ctrl+C to stop)")

    while True:
        # Read audio data from the input stream
        data = input_stream.read(CHUNK_SIZE)

        # Convert the raw audio data to a NumPy array
        audio_array = np.frombuffer(data, dtype=np.int16)

        # Calculate energy of the audio signal
        energy = np.sum(audio_array ** 2) / float(len(audio_array))

        # If energy is below the threshold, skip processing and output silence
        if energy < SILENCE_THRESHOLD:
            output_stream.write(b'\x00' * len(data))
            continue

        # Apply low-pass filter
        filtered_audio = signal.filtfilt(b, a, audio_array)

        shifted_audio = librosa.effects.pitch_shift(filtered_audio, sr=RATE, n_steps=int(np.log2(PITCH_SHIFT_FACTOR) * 12))

        # Apply distortion
        scaled_audio = shifted_audio / np.max(np.abs(shifted_audio))
        distorted_audio = np.tanh(DISTORTION * scaled_audio)

        # Add background noise
        noise = np.random.normal(scale=NOISE_SCALE * np.max(distorted_audio), size=len(distorted_audio))
        noisy_audio = distorted_audio + noise

        # Convert the processed audio data back to bytes
        processed_data = (noisy_audio * 32767).astype(np.int16).tobytes()

        # Play the processed audio data through the output stream
        output_stream.write(processed_data)
        #output_stream.write(data)

except KeyboardInterrupt:
    print("Simulation stopped.")

finally:
    # Close the streams and terminate PyAudio
    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    p.terminate()
