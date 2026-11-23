package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// wavFormat is what a WAV file's fmt chunk says about its samples.
type wavFormat struct {
	format, channels, rate, bits int
}

// WAV is a WAV file of integer or float PCM, its header read and checked, its samples still as bytes, so its length
// is known before any work is spent on them. Make one with ParseWAV.
type WAV struct {
	format  wavFormat
	samples []byte
	read    func([]byte) float32
}

// ParseWAV reads a WAV file's header. The WAV keeps data's samples without copying them, so data stays unchanged
// until Mono has read them.
func ParseWAV(data []byte) (WAV, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return WAV{}, errors.New("audio is not a WAV file")
	}

	f, samples, err := readWAVChunks(data)
	if err != nil {
		return WAV{}, err
	}

	if f.channels == 0 || f.rate == 0 || samples == nil {
		return WAV{}, errors.New("WAV file has no fmt or data chunk")
	}

	if f.channels > 8 {
		return WAV{}, fmt.Errorf("WAV file has %d channels", f.channels)
	}

	// An odd rate shares few factors with the model's, and its resampling filter would take gigabytes.
	if g := gcd(f.rate, SampleRate); f.rate < 8000 || f.rate > 192000 || f.rate/g*(SampleRate/g) > 1<<20 {
		return WAV{}, fmt.Errorf("unsupported sample rate %d", f.rate)
	}

	read, err := wavSampleReader(f)
	if err != nil {
		return WAV{}, err
	}
	return WAV{format: f, samples: samples, read: read}, nil
}

// Rate is the file's sample rate.
func (w WAV) Rate() int {
	return w.format.rate
}

// Seconds is how long the file's audio runs.
func (w WAV) Seconds() float64 {
	return float64(w.frames()) / float64(w.format.rate)
}

func (w WAV) frames() int {
	return len(w.samples) / (w.format.bits / 8 * w.format.channels)
}

// Mono is the file's audio as mono samples in [-1, 1] at its own rate, each frame's channels averaged.
func (w WAV) Mono() ([]float32, error) {
	f := w.format
	width := f.bits / 8
	out := make([]float32, w.frames())
	for i := range out {
		var sum float32
		for c := range f.channels {
			sum += w.read(w.samples[(i*f.channels+c)*width:])
		}

		// Only a float file can hold one, and a NaN or an infinity would spread to every model state that reads it.
		if math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
			return nil, errors.New("WAV file holds a sample that is not a number")
		}
		out[i] = sum / float32(f.channels)
	}
	return out, nil
}

// readWAVChunks reads the format and the sample bytes from the chunks after a WAV file's header.
func readWAVChunks(data []byte) (wavFormat, []byte, error) {
	var f wavFormat
	var samples []byte
	for at := 12; at+8 <= len(data); {
		// A chunk that claims more bytes than are left ends with the data, as a recording cut short leaves it.
		id, size := string(data[at:at+4]), binary.LittleEndian.Uint32(data[at+4:])
		body := data[at+8:][:min(uint64(size), uint64(len(data)-at-8))]
		switch id {
		case "fmt ":
			if len(body) < 16 {
				return wavFormat{}, nil, errors.New("WAV fmt chunk is too short")
			}
			f.format = int(binary.LittleEndian.Uint16(body))
			f.channels = int(binary.LittleEndian.Uint16(body[2:]))
			f.rate = int(binary.LittleEndian.Uint32(body[4:]))
			f.bits = int(binary.LittleEndian.Uint16(body[14:]))
			// WAVE_FORMAT_EXTENSIBLE names the real format in its subformat.
			if f.format == 0xfffe && len(body) >= 26 {
				f.format = int(binary.LittleEndian.Uint16(body[24:]))
			}
		case "data":
			samples = body
		}
		at += 8 + len(body) + len(body)%2
	}
	return f, samples, nil
}

// wavSampleReader is the function that reads one sample of the format as a value in [-1, 1].
func wavSampleReader(f wavFormat) (func([]byte) float32, error) {
	switch {
	case f.format == 1 && f.bits == 8:
		return func(b []byte) float32 { return (float32(b[0]) - 128) / 128 }, nil
	case f.format == 1 && f.bits == 16:
		return func(b []byte) float32 { return float32(int16(binary.LittleEndian.Uint16(b))) / 32768 }, nil
	case f.format == 1 && f.bits == 24:
		return func(b []byte) float32 {
			return float32(int32(uint32(b[0])<<8|uint32(b[1])<<16|uint32(b[2])<<24)>>8) / (1 << 23)
		}, nil
	case f.format == 1 && f.bits == 32:
		return func(b []byte) float32 { return float32(int32(binary.LittleEndian.Uint32(b))) / (1 << 31) }, nil
	case f.format == 3 && f.bits == 32:
		return func(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }, nil
	}
	return nil, fmt.Errorf("unsupported WAV encoding: format %d, %d bits", f.format, f.bits)
}

// AppendPCM16 appends audio to out as 16-bit little-endian samples, clipped to their range, so a stream can write span
// after span from one buffer. It scales by 32767, so 1 and -1 both fit, where reading divides by the type's full
// 32768, as soundfile does.
func AppendPCM16(out []byte, samples []float32) []byte {
	for _, v := range samples {
		s := math.Round(float64(v) * 32767)
		out = binary.LittleEndian.AppendUint16(out, uint16(int16(max(-32768, min(32767, s)))))
	}
	return out
}

// EncodeWAV writes samples at SampleRate as a WAV file of 16-bit mono PCM.
func EncodeWAV(samples []float32) []byte {
	pcm := AppendPCM16(nil, samples)
	return append(speechFormat.header(uint32(len(pcm))), pcm...)
}

// StreamingWAVHeader is the header of a WAV stream of 16-bit mono PCM at SampleRate whose length is not known yet. Its
// data size is the largest a header holds, as streaming writers write it, which players read as "until the stream
// ends".
func StreamingWAVHeader() []byte {
	return speechFormat.header(math.MaxUint32 - 36)
}

// speechFormat is the format of the speech the package writes.
var speechFormat = wavFormat{format: 1, channels: 1, rate: SampleRate, bits: 16}

// header is the header of a WAV file of format f whose samples take dataBytes.
func (f wavFormat) header(dataBytes uint32) []byte {
	b := append(make([]byte, 0, 44), "RIFF"...)
	b = binary.LittleEndian.AppendUint32(b, 36+dataBytes)
	b = append(b, "WAVEfmt "...)
	b = binary.LittleEndian.AppendUint32(b, 16)
	b = binary.LittleEndian.AppendUint16(b, uint16(f.format))
	b = binary.LittleEndian.AppendUint16(b, uint16(f.channels))
	b = binary.LittleEndian.AppendUint32(b, uint32(f.rate))
	b = binary.LittleEndian.AppendUint32(b, uint32(f.rate*f.channels*f.bits/8))
	b = binary.LittleEndian.AppendUint16(b, uint16(f.channels*f.bits/8))
	b = binary.LittleEndian.AppendUint16(b, uint16(f.bits))
	b = append(b, "data"...)
	return binary.LittleEndian.AppendUint32(b, dataBytes)
}
