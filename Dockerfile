# syntax=docker/dockerfile:1
FROM golang:1.27-alpine AS build

WORKDIR /src
COPY go.mod .
COPY cmd cmd
COPY internal internal
RUN CGO_ENABLED=0 GOEXPERIMENT=simd go build -o /2cent ./cmd/2cent

FROM alpine:3.22

WORKDIR /app

# A downloaded file is owned by root with mode 600, which the nobody user cannot read, and a folder ADD makes gets the
# file's mode, which no one can enter, so the folders come first.
RUN mkdir -p build/voices
ADD --chmod=644 --checksum=sha256:4a6633ae9abb0f6319a205c65cd5b555d3cedf0c2f2b242c9254196a9dfef225 \
    https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/2cent.safetensors build/2cent.safetensors
ADD --chmod=644 --checksum=sha256:7f0793b22136cc0cf1acde844012e03ba5aeddee17e24da534cdf4deabc8191c \
    https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/default.wav build/voices/default.wav
ADD --chmod=644 --checksum=sha256:e2f95c3395e91706a6212efe51fa9c7e1da7d8766be2db345ee2b4c983243534 \
    https://github.com/taylorchu/2cent-tts/releases/download/v0.7.0/default.txt build/voices/default.txt
COPY --from=build /2cent 2cent

USER nobody
EXPOSE 8080
ENTRYPOINT ["/app/2cent", "-listen", ":8080"]
