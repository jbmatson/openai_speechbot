import pyaudio
import wave
import webrtcvad
#mport requests
import os
import json
from openai import OpenAI

# A recommended way to store the key is in an environment variable. For this demo, I am storing in a file.
key_location = 'C:\speechbot\openai_apikey.txt'

with open(key_location, 'r') as file:
   api_key = file.readline().strip()

openai = OpenAI(
   api_key = api_key
)

# Initialize the VAD (Voice Activity Detector)
vad = webrtcvad.Vad()
vad.set_mode(3)  # You can experiment with different modes (1-3) for sensitivity

# Configure the audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000         # Samples per second
CHUNK = 320          # chunk size
RECORD_SECONDS = 10  # Maximum length of each audio segment
WAVE_OUTPUT_FILENAME = "audio.wav"
SILENCE_LIMIT = 2    # Number of seconds of silence/non-speech before we assume user has stopped talking

# WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz. A frame must be either 10, 20, or 30 ms in duration.
# For example, if your sample rate is 16000 Hz, then the only allowed frame/chunk sizes are 16000 * ({10,20,30} / 1000) = 160, 320 or 480 samples. 
# Since each sample is 2 bytes (16 bits), the only allowed frame/chunk sizes are 320, 640, or 960 bytes.

# Create a PyAudio stream for recording
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize variables for audio data
frames = []
audio_data = b""
in_speech = False
silence_threshold = 0

print("Listening... (Press Ctrl+C to stop)")
while True:
   try:
      audio_chunk = stream.read(CHUNK)
      frames.append(audio_chunk)
      audio_data += audio_chunk

      is_speech = vad.is_speech(audio_chunk, RATE)

      if is_speech:
         if not in_speech:
                  # if speech is detected but script not aware of speech
                  # make it aware
                  in_speech = True
                  silence_threshold = 0
         continue
      elif not is_speech and in_speech:
         # check if user has been quiet long enough to assume they have stopped talking
         if silence_threshold < SILENCE_LIMIT * (RATE / CHUNK):
            # not yet - wait longer
            silence_threshold += 1
            continue
         else:
            # silence has reached limit - process and send the recording for transcription
            in_speech = False
            if audio_data:
               # Save the audio segment to a WAV file
               with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                  wf.setnchannels(CHANNELS)
                  wf.setsampwidth(audio.get_sample_size(FORMAT))
                  wf.setframerate(RATE)
                  wf.writeframes(audio_data)
                  wf.close()
                  
               # Send the audio segment to OpenAI Whisper for speech recognition
               audio_url = "https://api.openai.com/v1/audio/transcriptions"
               audio_file = open(WAVE_OUTPUT_FILENAME, 'rb')

               try:
                  transcription = openai.audio.transcriptions.create(
                     model = "whisper-1", 
                     file = audio_file
                  )
               except openai.APIError as e:
                  print(f"OpenAI API returned an API Error: {e}")
                  pass
               except openai.APIConnectionError as e:
                  print(f"Failed to connect to OpenAI API: {e}")
                  pass
               except openai.RateLimitError as e:
                  print(f"OpenAI API request exceeded rate limit: {e}")
                  pass

               print("Transcription:", transcription.text)
               audio_file.close()
               
               # Remove the temporary WAV file
               os.remove(WAVE_OUTPUT_FILENAME)
                  
               audio_data = b""

               # TODO: Send the transcription text to ChatGPT to get a response

               # TODO: Send the response to a text-to-speect service, stream the response to audio output


   except KeyboardInterrupt:
      break

# Cleanup
stream.stop_stream()
stream.close()
audio.terminate()