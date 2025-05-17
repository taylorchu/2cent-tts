# 2cent-tts

This experiment aims to develop an open-weight, cost-effective speech model using just a few hundred hours of synthetic training data. After training, the model is converted to [gguf](https://github.com/ggml-org/llama.cpp) for inference purposes.

Currently, the system only supports US English, though support for additional languages is planned for future releases.

## Samples

| Index | Sample Path                                                                         |
| ----- | ----------------------------------------------------------------------------------- |
| 0     | [](https://github.com/user-attachments/assets/aac168bd-214d-4e78-a7f0-24679ab5e084) |
| 1     | [](https://github.com/user-attachments/assets/c758c0aa-8df5-4827-a72a-d58780031dd2) |
| 2     | [](https://github.com/user-attachments/assets/c0ef39d5-bbdc-47c1-86db-0fac096841b3) |
| 3     | [](https://github.com/user-attachments/assets/de9bd205-1981-452f-8110-52b11d2b04ce) |
| 4     | [](https://github.com/user-attachments/assets/790c3622-8af5-4e7a-b581-53e050062b7e) |
| 5     | [](https://github.com/user-attachments/assets/b54a7445-08ed-4532-8517-21fb7b02c27b) |
| 6     | [](https://github.com/user-attachments/assets/1ffbd093-37df-4406-ba4c-1e53146b639d) |

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
