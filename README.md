# 2cent-tts

This experiment aims to develop an open-weight, cost-effective speech model. The model runs on the dependency-free Go
inference engine in [cmd/2cent](cmd/2cent) and [internal](internal).

Currently, the system only supports US English, though support for additional languages is planned for future releases.

## Features

- Small model (82M parameters, plus the SNAC codec)
- Voice cloning from a 1.5 to 10 s reference clip and its transcript
- High-quality 24kHz audio output, streamed out as it is made
- Sound tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<gasp>`, `<cough>`, `<groan>`, `<clearing_throat>`
- Text normalization for numbers, dates, times, money, units, phone numbers, URLs, email addresses and Markdown

## Demo

https://github.com/user-attachments/assets/aa0bd359-d2fd-48f2-b89c-cb29ceba2eef

## Performance

Measured on a laptop CPU with the release build (`GOEXPERIMENT=simd`) and the default voice, model loading included.
Real-time factor is seconds of audio per second of wall time.

| Threads (`GOMAXPROCS`) | ~19 s paragraph | ~53 s passage |
| --- | --- | --- |
| all | 3.5x real time | 3.3x |
| 4 | 3.0x | 2.7x |
| 2 | 1.8x | 1.7x |

Memory stays under 1 GB. Past about four threads, speed grows little: each step reads the weights from memory, and
memory, not the core count, sets the limit. As a server, with the model loaded, the first audio of a request arrives in
0.17 s at all threads, 0.22 s at 4 and 0.37 s at 2.

## Benchmarks

Each line is spoken in the voice of a reference clip by the models that clone, and 2cent never trained on those voices
or lines. Parakeet transcribes the speech to count wrong words.

| Model | Size | Cloning | Sound tags | Wrong words | Seed-TTS voices | Rare words | Failed lines | Tags heard | UTMOS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2cent v0.7.0 | 102M | yes | yes | 1.4% | 1.2% | 6.4% | 0.0% | 83% | 4.22 |
| Pocket TTS 18M | 28M | yes | no | 1.0% | 1.3% | 6.8% | 0.9% | - | 4.29 |
| BlueTTS v3 | 97M | no | no | 1.1% | 1.1% | 5.5% | 1.4% | - | 4.48 |
| Pocket TTS 100M | 110M | yes | no | 1.0% | 1.1% | 5.7% | 0.8% | - | 4.34 |
| MOSS-TTS Nano | 165M | yes | no | 4.2% | 4.4% | 8.2% | 6.3% | - | 4.10 |
| Audio8 TTS 0.1B | 507M | yes | no | 1.0% | 1.2% | 5.4% | 0.3% | - | 4.46 |
| Chatterbox Turbo | 746M | yes | yes | 1.4% | 2.2% | 6.9% | 0.6% | 41% | 4.36 |
| OmniVoice | 814M | yes | yes | 1.2% | 1.2% | 4.8% | 2.5% | 19% | 4.31 |
| VibeVoice Realtime 0.5B | 1,018M | no | no | 1.7% | 1.7% | 7.1% | 0.5% | - | 4.41 |
| Qwen3-TTS 0.6B | 1,085M | yes | no | 1.3% | 1.2% | 6.5% | 4.1% | - | 4.44 |
| CosyVoice3 0.5B | 1,244M | yes | yes | 1.5% | 2.1% | 6.7% | 2.2% | 6% | 4.32 |
| Maya1 | 3,321M | no | yes | 1.1% | 1.0% | 5.3% | 0.6% | 37% | 4.43 |

- Size: parameters of every network needed to speak, the codec or vocoder and the reference clip's encoders included.
- Cloning and sound tags: whether the model takes a reference clip, and inline tags such as `<laugh>`.
- Wrong words: 157 voices, one line each, with clean reference clips.
- Seed-TTS voices: wrong words on 300 lines of Seed-TTS-eval's English set, with its reference clips.
- Rare words: wrong words on 200 lines holding words that are rare in training text.
- Failed lines: about 1,000 long paragraphs, short replies and lines with sound tags; a line fails when the speech
  runs on past the text, skips a sentence or loops.
- Tags heard: lines with a sound tag where a sound detector hears it against the same line spoken without the tag.
- UTMOS predicts how natural a listener finds the audio, from 1 to 5.

## Usage

### Run a release binary

Download the binary for your system from the
[v0.7.0 release](https://github.com/taylorchu/2cent-tts/releases/tag/v0.7.0): `2cent-linux-amd64`,
`2cent-linux-arm64`, `2cent-darwin-arm64` or `2cent-windows-amd64.exe`. Put the model and the default voice in `build/`
next to it:

```sh
curl -L -o 2cent https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/2cent-darwin-arm64
chmod +x 2cent
mkdir -p build/voices
curl -L -o build/2cent.safetensors https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/2cent.safetensors
curl -L -o build/voices/default.wav https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/default.wav
curl -L -o build/voices/default.txt https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/default.txt

# Speak a text in the default voice
./2cent -text "Hello there. <laugh> That was fun." -voice default -o out.wav

# Clone a voice from a clip and what it says; to try it, clone the default voice from its own clip
./2cent -text "Hello there." -ref-wav build/voices/default.wav -ref-text "$(cat build/voices/default.txt)" -o out.wav

# Serve the speech API
./2cent -listen :8080
```

To build from source instead, install Go 1.27 and run
`GOEXPERIMENT=simd go install github.com/taylorchu/2cent-tts/cmd/2cent@v0.7.0`, which puts `2cent` in
`$(go env GOPATH)/bin`, or from a clone, `GOEXPERIMENT=simd go build ./cmd/2cent`. A build without `GOEXPERIMENT=simd`
also works, at less than half the speed.

Run `2cent` from the folder that holds `build/`, or pass `-model` and `-voices` with full paths. `-text` needs `-voice`
or `-ref-wav`. `-seed` fixes sampling. `TWOCENT_SETTINGS` takes a JSON object of release settings to override, such as
`{"temperature": 0.5}`, for measuring the model with other choices.

A reference clip is a WAV file holding 1.5 to 10 seconds of speech once its leading and trailing silence is trimmed.
Its text must say what the clip says, word for word: nothing checks it, and a wrong text makes worse speech. To add a
voice, put `name.wav` and `name.txt` in `build/voices`; the server reads its voices when it starts and speaks with
this one as `"voice": "name"`.

### Run with Docker

```sh
# Build an image named 2cent from the repository, with the release's model and default voice
docker build -t 2cent .

# Serve the speech API on port 8080; on docker stop, the requests in flight get up to a minute to finish
docker run -p 8080:8080 --stop-timeout 60 2cent
```

### Call the API

These examples work with `./2cent -listen :8080` and with the Docker container. They play audio with `ffplay`, part
of ffmpeg.

```sh
# Speech as PCM, played as it streams
cat <<EOF | curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d @- \
  --output - | ffplay -f s16le -ar 24000 -i -
{
  "model": "tts-1",
  "input": "Hello, this is a test of text to speech.",
  "voice": "default",
  "response_format": "pcm"
}
EOF

# Speech as WAV, played as it streams
cat <<EOF | curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d @- \
  --output - | ffplay -
{
  "model": "tts-1",
  "input": "Hello, this is a test of text to speech.",
  "voice": "default",
  "response_format": "wav"
}
EOF

# Speech saved as output.wav
cat <<EOF | curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d @- \
  --output output.wav
{
  "model": "tts-1",
  "input": "Hello, this is a test of text to speech.",
  "voice": "default",
  "response_format": "wav"
}
EOF

# A cloned voice, from a clip and what it says, saved as cloned_output.wav
cat <<EOF | curl -X POST http://localhost:8080/v1/audio/voice-cloning \
  -H "Content-Type: application/json" \
  -d @- \
  --output cloned_output.wav
{
  "model": "tts-1",
  "input": "Hello, this is a test of text to speech.",
  "audio_text": "$(cat build/voices/default.txt)",
  "audio": "$(base64 < build/voices/default.wav | tr -d '\n')",
  "response_format": "wav"
}
EOF
```

OpenAI's clients work with `base_url` set to `http://localhost:8080/v1`. They ask for mp3 unless told otherwise, so
pass `response_format` `"wav"` or `"pcm"`, and a voice from `build/voices`: OpenAI's own voice names, such as
`alloy`, are refused.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
client.audio.speech.create(model="tts-1", voice="default", input="Hello there.", response_format="wav") \
    .write_to_file("hello.wav")
```

### HTTP API

- `POST /v1/audio/speech` takes OpenAI's speech request. `input` holds up to 4096 characters and `voice` names a
  voice. Speech streams out as it is made, as 24 kHz 16-bit mono: `response_format` is `pcm`, the default, or `wav`,
  the same samples behind a WAV header whose length reads "until the stream ends". `speed` must be 1, and `model` is
  ignored.
- `POST /v1/audio/voice-cloning` takes the same request with `audio`, a base64 WAV reference clip, and `audio_text`,
  what it says, in place of `voice`. Each endpoint refuses the other's voice fields.
- `GET /health` answers once the model is loaded.

Tags of lowercase letters other than the sound tags above, such as `<br>` or `</b>`, are dropped from the text. A
request whose text or reference clip fails a check gets a 400. The text and the clip's WAV header are checked at once;
a clip with no speech, or too little or too much, is refused once the request has a slot. The server speaks for
`-max-requests` requests at once; a request waits up to `-queue-wait` for a slot, then gets a 503 with `Retry-After`. A
request that finds as many requests already waiting gets the 503 at once. Other OpenAI fields, such as
`instructions`, are ignored.

### Go tests

The tests compare the Go engine with fixtures that the training code writes and that are not published: the release
weights, the golden values PyTorch computed from them, and the normalizer corpus. The release settings, such as the
guidance scale, the temperature and the chunk limits, are stored inside the weights file.

| Variable | Fixture | Default |
| --- | --- | --- |
| `TWOCENT_WEIGHTS` | release weights | `build/2cent.safetensors` |
| `TWOCENT_GOLDEN` | golden values | `build/golden.json.gz` |
| `TWOCENT_NORMALIZE_GOLDEN` | normalizer corpus | `build/normalize_golden.json.gz` |

A test skips when its fixture is missing. Set `TWOCENT_REQUIRE_FIXTURES=1` to make it fail instead. The release
build uses the SIMD kernels, which need Go 1.27 and `GOEXPERIMENT=simd`, so run the suite both ways. The worker pool
runs on several goroutines and requests run side by side, so the tests also run under the race detector, on two
threads, the fewest a pool runs with, and on ten, past the eight that take small calls. `deadcode` lists the functions
only tests reach, and prints nothing when there are none:

```sh
export TWOCENT_REQUIRE_FIXTURES=1
go test ./...
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -race -timeout 30m ./internal/tensor ./internal/audio ./internal/engine ./cmd/2cent
GOMAXPROCS=2 GOEXPERIMENT=simd go test ./internal/tensor
GOMAXPROCS=10 GOEXPERIMENT=simd go test ./internal/tensor
GOEXPERIMENT=simd go test -run '^$' -bench . ./internal/engine ./internal/audio
GOEXPERIMENT=simd go run golang.org/x/tools/cmd/deadcode@v0.53.0 ./cmd/2cent
GOEXPERIMENT=simd golangci-lint run ./...
```

## Technical Implementation Details

### Tokenization Strategy

The tokenizer is a Hugging Face `tokenizer.json`, which the Go engine reads and runs itself. Its vocabulary is
byte-level BPE text tokens and SNAC audio tokens, written `<audio_X>`, such as `<audio_1024>`.

### Generation Pipeline

The model reads a prompt whose format depends on its version, then writes audio tokens, `<audio_X><audio_X>...</s>`:

v0.1.0 and v0.2.0: `<ipa_X><ipa_X>...<s>` format

v0.3.0: `<s><speaker><speaker_X><speaker_X>...<text><ipa_X><ipa_X>...<generate>` format, which includes speaker tokens,
followed by the IPA sequence

v0.4.0: `<s><speaker><speaker_X><speaker_X>...<emotion><emotion_X><emotion_X>...<text><ipa_X><ipa_X>...<generate>`
format, which includes emotion tokens on top of the previous version.

v0.5.0: Use the same format as v0.4.0.

v0.6.0: Use the same format as v0.4.0. Additionally, `<s><text><ipa_X><ipa_X>...<generate><audio>...` format, which
includes sampled audio for voice cloning.

Before v0.7.0, the IPA came from espeak-ng, and a `.` separated the phonetic segments at a pause, such as a comma.

v0.7.0: `<s><text>{reference text}<audio>{reference audio}<text>{whole text}<audio>{first two words}` format, with raw
text in place of IPA and a reference clip in place of speaker and emotion tokens. The model writes two SNAC frames at a
time and says after each whether a word or tag started, and the next word of the text is fed then, so the text stays two
words ahead of the speech; two `</s>` close the text, and the start that finds no word left ends the speech. Each chunk
after the first opens with the last words of the one before and reads their audio before it writes, so the speech
carries on across the join.

### Hierarchical Token Structure

A frame's seven audio tokens are laid out depth first. For example, in a sequence `[1,2,3,4,5,6,7]`, token 1 represents
the coarsest level of audio representation, tokens 2 and 5 are the children of token 1, tokens 3 and 4 are the children
of token 2, and tokens 6 and 7 are the children of token 5.

### Audio Synthesis

The SNAC decoder turns the three levels of codes into 24 kHz audio.

## Responsible Use

Clone only voices you have permission to use, and do not use the model to impersonate or deceive.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

The release weights include the [SNAC](https://github.com/hubertsiuzdak/snac) codec and the
[Silero VAD](https://github.com/snakers4/silero-vad) speech detector, both under the MIT License.
