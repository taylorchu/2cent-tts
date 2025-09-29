# 2cent-tts

This experiment aims to develop an open-weight, cost-effective speech model using ~10k hours of synthetic training data. After training, the model is converted to [gguf](https://github.com/ggml-org/llama.cpp) for inference purposes.

Currently, the system only supports US English, though support for additional languages is planned for future releases.

## Features

- Small model (under 100M parameters)
- Thousands of voices available
- Voice cloning
- Emotion support
- Multi-turn dialogue in single inference
- High-quality 24kHz audio output

## Samples

https://github.com/user-attachments/assets/1272c11e-8026-47b2-aa3a-a8bb7de0b0d0

https://github.com/user-attachments/assets/d6733aa2-de66-4d45-885f-3bcfe3c2fc90

## Testing

```
# Build a Docker image named '2cent' from the current directory
docker build -t 2cent .

# Run the Docker container, mapping port 8080 on the host to port 80 in the container
docker run -p 8080:80 2cent

# Send a text-to-speech request to the API and play the audio response
# - Makes a POST request to the speech endpoint
# - Specifies tts-1 model with text input
# - Requests PCM audio format
# - Pipes the output to ffplay for immediate playback
curl http://localhost:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"tts-1","input":"Hello, this is a test of text to speech.","voice":"<speaker><speaker_35><speaker_80><speaker_108><speaker_119><speaker_18><speaker_44><emotion><emotion_99><emotion_114><emotion_22><emotion_25><emotion_126><emotion_1>","response_format":"pcm"}' --output - | ffplay -f s16le -ar 24000 -ac 1 -

Or WAV audio format
curl http://localhost:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"tts-1","input":"Hello, this is a test of text to speech.","voice":"<speaker><speaker_35><speaker_80><speaker_108><speaker_119><speaker_18><speaker_44><emotion><emotion_99><emotion_114><emotion_22><emotion_25><emotion_126><emotion_1>","response_format":"wav"}' --output - | ffplay -

# Or Saves the result as "output.wav" in the current directory
curl http://localhost:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"tts-1","input":"Hello, this is a test of text to speech.","voice":"<speaker><speaker_35><speaker_80><speaker_108><speaker_119><speaker_18><speaker_44><emotion><emotion_99><emotion_114><emotion_22><emotion_25><emotion_126><emotion_1>","response_format":"wav"}' --output output.wav
```

## Technical Implementation Details

> [!NOTE]
> We're currently focusing our support on the provided inference code, which represents our recommended implementation.
> While we aren't offering alternative methods at this time, you're welcome to create your own implementation using the Technical Implementation Details as a guide.

### Tokenization Strategy

The system employs huggingface tokenizers for processing both audio and International Phonetic Alphabet (IPA) representations. Our vocabulary consists of SNAC audio tokens combined with additional IPA tokens. Audio tokens follow a standardized notation format of `<audio_X>`, where X represents the token identifier (e.g., `<audio_1024>`).

### Phonetic Representation

The IPA representations are derived directly from espeak-ng, ensuring consistent phonetic transcription across languages. When encountering pauses in speech (such as those indicated by commas in text), we use the "." character as a delimiter to separate multiple phonetic segments. This approach provides a clean separation between distinct phonetic units while preserving the natural rhythm of speech.

### Generation Pipeline

The tokenized input is processed with different structured prefix formats depending on the model version:

v0.1.0 and v0.2.0: `<ipa_X><ipa_X>...<s>` format, where each IPA token is properly tagged and sequenced

v0.3.0: `<s><speaker><speaker_X><speaker_X>...<text><ipa_X><ipa_X>...<generate>` format, which includes speaker tokens, followed by the IPA sequence

This formatted input prompts the model to generate a corresponding sequence of audio tokens in the form `<audio_X><audio_X>...</s>`. The inclusion of speaker tokens in v0.3.0 provides additional context for audio synthesis, while the standardized input-output pattern enables consistent audio generation across various inputs and model versions.

v0.4.0: `<s><speaker><speaker_X><speaker_X>...<emotion><emotion_X><emotion_X>...<text><ipa_X><ipa_X>...<generate>` format, which includes emotion tokens on top of the previous version.

### Hierarchical Token Structure

The generated audio tokens are converted to sequential token IDs (1, 2, 3, etc.) organized in a depth-first hierarchical structure. For example, in a sequence `[1,2,3,4,5,6,7]`, token 1 represents the coarsest level of audio representation, while tokens 2 and 5 function as leaf nodes of token 1. Similarly, tokens 3 and 4 are leaf nodes of token 2, while tokens 6 and 7 are leaf nodes of token 5. This hierarchical approach enables multi-resolution audio modeling.

### Audio Synthesis

The final stage involves converting these hierarchically structured audio tokens, which contain three levels of granularity, through the SNAC decoder to produce the actual audio waveform. This multi-level approach allows for precise control over the acoustic properties of the synthesized speech, resulting in more natural-sounding output.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
