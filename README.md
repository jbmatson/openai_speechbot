# openai_speechbot

### Prerequisites:
The following python packages must be installed:
* pyaudio
* openai
* webrtcvad  (may require the Microsoft C++ Build Tools to be installed for Windows computers )
* openai
* elevenlabs

*mpv* must be installed for ElevenLabs streaming audio output to work. Install from here: https://mpv.io/

On Windows you may need to add the install folder to the path for the ElevenLabs library to be able to find it.

### Ideas for future projects

* Call into a customised VoiceFlow flow instead of using ChatGPT  - make a simple emergency services call operator flow to use
* Instead of using a Python script, build it as a simple Django website. Would need to investigate if/how it would be possible to do the voice recording and voice activity detection in the web client (WebRTC VAD?), stream the voice recording to the server, then stream the voice response back to the client