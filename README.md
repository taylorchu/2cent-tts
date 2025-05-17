# 2cent-tts

This experiment aims to develop an open-weight, cost-effective speech model using just a few hundred hours of synthetic training data. After training, the model is converted to [gguf](https://github.com/ggml-org/llama.cpp) for inference purposes.

Currently, the system only supports US English, though support for additional languages is planned for future releases.

## Samples

| Index | Sample Path       |
| ----- | ----------------- |
| 0     | [](samples/0.mp4) |
| 1     | [](samples/1.mp4) |
| 2     | [](samples/2.mp4) |
| 3     | [](samples/3.mp4) |
| 4     | [](samples/4.mp4) |
| 5     | [](samples/5.mp4) |
| 6     | [](samples/6.mp4) |

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
curl http://localhost:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"tts-1","input":"Hello, this is a test of text to speech.","response_format":"pcm"}' --output - | ffplay -f s16le -ar 24000 -ac 1 -

# Or Saves the result as "output.wav" in the current directory
curl http://localhost:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"tts-1","input":"Hello, this is a test of text to speech.","response_format":"pcm"}' --output - | ffmpeg -f s16le -ar 24000 -ac 1 -i - output.wav
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
