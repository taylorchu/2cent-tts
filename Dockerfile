FROM alpine AS onnxruntime

ADD https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz /
RUN tar -xvf /onnxruntime-linux-x64-1.22.0.tgz

FROM alpine AS tts-http-server

ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.5.0/tts-http-server-linux-amd64.zip /
RUN unzip /tts-http-server-linux-amd64.zip

FROM ghcr.io/ggml-org/llama.cpp:server-b6783

RUN \
  apt-get update \
  && apt-get install -y --no-install-recommends --no-install-suggests \
  espeak-ng \
  && rm -rf /var/lib/apt/lists/*

COPY --from=onnxruntime /onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so libonnxruntime.so
ENV ONNX_PATH=/app/libonnxruntime.so

ADD https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx snac.onnx
ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.5.0/2cent.gguf 2cent.gguf
ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.5.0/tokenizer.json tokenizer.json
COPY --from=tts-http-server /tts-http-server-linux-amd64 tts-http-server

RUN chmod +x tts-http-server

ENV PATH="$PATH:/app"
ENV LLAMA_SERVER_ARGS="--samplers temp"
ENV IPA_EXCLUDE_TAGS="<breath>,<clearing_throat>,<gasp>,<growl>,<grunt>,<heavy_breathing>,<hiss>,<hum>,<kiss>,<laugh>,<moan>,<mouth_sound>,<purr>,<scoff>,<shushing>,<sigh>,<sniff>,<sob>,<tongue_click>,<whisper>,<yawn>,<unknown_sound>"

ENTRYPOINT ["tts-http-server"]
