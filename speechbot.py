"""
This is a simple chat bot with that uses webrtvad for voice recording, 
OpenAI whisper transcription, ChatGPT, and Whisper.
"""
import wave
import os
import pyaudio
import webrtcvad
from openai import OpenAI
from elevenlabs import generate, stream, set_api_key, APIError, RateLimitError, AuthorizationError

# Configure the audio recording parameters

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000                    # Samples per second
CHUNK = 320                     # chunk size in bytes
RECORD_SECONDS = 10             # Maximum length of each audio segment
WAVE_OUTPUT_FILENAME = "audio.wav"
VAD_MODE_HIGH_SENSITIVITY = 3
SILENCE_LIMIT = 2               # Number of seconds of silence/non-speech before
                                #   we assume user has stopped talking

# NOTE: WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
# A frame must be either 10, 20, or 30 ms in duration.
# For example, if your sample rate is 16000 Hz, then the only allowed frame/chunk sizes are
# 16000 * ({10,20,30} / 1000) = 160, 320 or 480 samples.
# Since each sample is 2 bytes (16 bits), the only allowed frame/chunk sizes are
# 320, 640, or 960 bytes.

# Configure GPT parameters
GPT_MODEL = "gpt-3.5-turbo"
GPT_CHAT_TEMPERATURE = 0.7


def main():
    """
    This is the main function for the SpeechBot script
    """

    # Get API keys
    # The recommended way to store the keys is in environment variables.
    # For this demo, I am storing them in files.

    openai_api_key = get_api_key_from_file('C:\\speechbot\\openai_apikey.txt')
    openai = OpenAI(
        api_key = openai_api_key
    )

    elevenlabs_api_key = get_api_key_from_file('C:\\speechbot\\elevenlabs_apikey.txt')
    set_api_key(elevenlabs_api_key)

    # Initialize the VAD (Voice Activity Detector)
    vad = webrtcvad.Vad()
    vad.set_mode(VAD_MODE_HIGH_SENSITIVITY)  # You can experiment with different modes
                                             # (1-3) for sensitivity

    # Create a PyAudio stream for recording
    audio = pyaudio.PyAudio()
    rec_stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

    # Initialize variables for audio data
    frames = []
    audio_data = b""
    in_speech: bool = False
    silence_threshold: int = 0
    sample_width: int = audio.get_sample_size(FORMAT)

    # Initialise variables for chat conversation
    said_goodbye: bool = False
    chat_messages = []   # the chat message history

    speak_string("assistant", "Hello, how can I help you?", chat_messages)

    print("Listening... (Press Ctrl+C to stop or say 'goodbye')")
    while True and not said_goodbye:
        try:
            audio_chunk = rec_stream.read(CHUNK)
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
                    save_audio_to_wave(WAVE_OUTPUT_FILENAME, sample_width, audio_data)

                    # Send the audio segment to OpenAI Whisper for speech recognition
                    transcription: str = transcribe_audio(WAVE_OUTPUT_FILENAME, openai)

                    # Remove the temporary WAV file
                    os.remove(WAVE_OUTPUT_FILENAME)
                    audio_data = b""

                    chat_messages.append({ "role": "user", "content": transcription })

                    # Send the transcription text to ChatGPT to get a response
                    response_text = get_chatgpt_response(chat_messages, openai)

                    # Send the response to ElevenLabs text-to-speect service,
                    # stream the response to audio output
                    speak_string("assistant", response_text, chat_messages)

                    if transcription.lower() == "goodbye.":
                        said_goodbye = True
                    else:
                        print("Listening... (Press Ctrl+C to stop or say 'goodbye')")

        except KeyboardInterrupt:
            break

    # Cleanup
    rec_stream.stop_stream()
    rec_stream.close()
    audio.terminate()

    # End of main function


def speak_string(role: str, text: str, chat_messages):
    """
    Speak a string using the ElevenLabs text-to-speech streaming API, 
    adds it to the chat messages list, and prints to terminal for the given role.

    :param role: The role of the speaker
    :param text: The text to speak
    :param chat_messages: The chat message list containing the whole conversation
    """
    chat_messages.append({"role": role, "content": text})

    print(role + ": " + text)

    try:
        audio_stream = generate(
            text = text,
            voice = "Rachel",
            model = "eleven_multilingual_v2",
            stream=True
        )
        stream(audio_stream)

    except RateLimitError as e:
        print("ElevenLabs rate limit error - " + e.message)
    except AuthorizationError as e:
        print("ElevenLabs authorization error - " + e.message)
    except APIError as e:
        print("ElevenLabs API error - " + e.message)

def save_audio_to_wave(wave_file_path, sample_width, audio_data):
    """
    Saves the audio data to the given path

    :param wave_file_path: The file path to save the wave file to
    :param sample_width: The sample width in bytes to use
    :param audio_data: The frames of audio data to write to the file
    """
    try:
        wf: wave.Wave_write
        with wave.open(wave_file_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
            wf.close()
    except wave.Error as e:
        print("Error writing temporary wave file - " + e)

def transcribe_audio(wave_file_path: str, openai: OpenAI) -> str:
    """ Transcribes the given wave file using the OpenAI audio transcriptions (Whisper) API

        :param wave_filename: The path to the wave file to transcribe
        :param openai: The OpenAI library object to use for transcription
    """
    transcription: str = ""
    with open(wave_file_path, 'rb') as audio_file:
        try:
            transcription_result = openai.audio.transcriptions.create(
                model = "whisper-1",
                file = audio_file
            )
            transcription = transcription_result.text
            print("Transcription:", transcription)
            audio_file.close()

        except openai.OpenAIError as e:
            print(e.http_status)
            print(e.error)

    return transcription

def get_chatgpt_response(chat_messages, openai: OpenAI) -> str:
    """ 
    Sends the chat message list including latest entry to ChatGPT chat completions 
    API to get a response from the AI

    :param chat_messages: The chat messages list (history)
    :param openai: The OpenAI library object to use
    """
    response = openai.chat.completions.create(
        model = GPT_MODEL,
        messages = chat_messages,
        temperature = GPT_CHAT_TEMPERATURE
    )

    response_text = response.choices[0].message.content
    return response_text

def get_api_key_from_file(key_file_path: str) -> str:
    """
    Gets an API key from the given file location

    :param key_file_path: The path to the API key file
    """
    api_key: str = ""

    try:
        with open(key_file_path, 'r') as file:
            api_key = file.readline().strip()
    except OSError:
        print("Could not open/read file: " + key_file_path)

    return api_key


if __name__ == "__main__":
    main()
