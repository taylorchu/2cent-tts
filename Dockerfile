FROM alpine AS onnxruntime

ADD https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz /
RUN tar -xvf /onnxruntime-linux-x64-1.22.0.tgz

FROM ghcr.io/ggml-org/llama.cpp:server-b5318

RUN \
  apt-get update \
  && apt-get install -y --no-install-recommends --no-install-suggests \
  espeak-ng \
  && rm -rf /var/lib/apt/lists/*

COPY --from=onnxruntime /onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so libonnxruntime.so
ENV ONNX_PATH=/app/libonnxruntime.so

ADD https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx snac.onnx
ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.3.0/2cent.gguf 2cent.gguf
ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.3.0/tokenizer.json tokenizer.json
ADD https://github.com/taylorchu/2cent-tts/releases/download/v0.3.0/tts-http-server tts-http-server

RUN chmod +x tts-http-server

ENV PATH="$PATH:/app"
ENV LLAMA_SERVER_ARGS="--samplers temp"

ENTRYPOINT ["tts-http-server"]
