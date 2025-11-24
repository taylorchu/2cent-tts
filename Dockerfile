FROM alpine AS onnxruntime

ADD https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz /
RUN tar -xvf /onnxruntime-linux-x64-1.23.2.tgz

FROM alpine AS tts-http-server

# ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.6.0/tts-http-server-linux-amd64.zip /
COPY build/tts-http-server-linux-amd64.zip /
RUN unzip /tts-http-server-linux-amd64.zip

FROM alpine AS ffmpeg

ADD https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz /
RUN tar -xf /ffmpeg-release-amd64-static.tar.xz && \
  mv /ffmpeg-*-amd64-static /ffmpeg-static

FROM ghcr.io/ggml-org/llama.cpp:server-b7131

RUN \
  apt-get update \
  && apt-get install -y --no-install-recommends --no-install-suggests \
  espeak-ng \
  && rm -rf /var/lib/apt/lists/*

COPY --from=onnxruntime /onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so libonnxruntime.so
ENV ONNX_PATH=/app/libonnxruntime.so

ADD https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx snac.onnx
ADD https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/encoder_model.onnx snac_encoder.onnx
# ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.6.0/2cent.gguf 2cent.gguf
# ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.6.0/tokenizer.json tokenizer.json
COPY build/2cent.gguf 2cent.gguf
COPY build/tokenizer.json tokenizer.json
COPY --from=tts-http-server /tts-http-server-linux-amd64 tts-http-server
COPY --from=ffmpeg /ffmpeg-static/ffmpeg /usr/local/bin/ffmpeg

RUN chmod +x tts-http-server

ENV PATH="$PATH:/app"
ENV LLAMA_SERVER_ARGS="--samplers temp"
ENV IPA_EXCLUDE_TAGS="<breath>,<clearing_throat>,<gasp>,<growl>,<grunt>,<heavy_breathing>,<hiss>,<hum>,<kiss>,<laugh>,<moan>,<mouth_sound>,<purr>,<scoff>,<shushing>,<sigh>,<sniff>,<sob>,<tongue_click>,<whisper>,<yawn>,<unknown_sound>"

ENTRYPOINT ["tts-http-server"]
