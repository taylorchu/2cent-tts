package audio

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"strconv"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// TestResample resamples a mix of tones between the rates references come in and matches torchaudio's output.
func TestResample(t *testing.T) {
	g := fixtures.LoadGolden(t)
	var n int
	if err := json.Unmarshal(g.Resample["input_samples"], &n); err != nil {
		t.Fatal(err)
	}

	input := make([]float32, n)
	for k := range input {
		x := float64(k)
		input[k] = float32(0.5*math.Sin(0.013*x) + 0.25*math.Sin(0.41*x+1.0) + 0.1*math.Sin(2.3*x))
	}

	for _, rates := range [][2]int{{48000, 24000}, {24000, 16000}, {44100, 24000}, {16000, 24000}} {
		var encoded string
		if err := json.Unmarshal(g.Resample[strconv.Itoa(rates[0])+"_"+strconv.Itoa(rates[1])], &encoded); err != nil {
			t.Fatal(err)
		}

		got, want := Resample(input, rates[0], rates[1]), fixtures.Floats(t, encoded)
		if len(got) != len(want) {
			t.Errorf("%v: %d samples, want %d", rates, len(got), len(want))
			continue
		}

		// One windowed sinc in float32, summed in another order than torchaudio's.
		if d := approx.MaxDifference(got, want); d > 1e-5 {
			t.Errorf("%v: samples differ by up to %g", rates, d)
		}
	}
}

// wavFile is a PCM WAV file of data, its header holding the format given.
func wavFile(channels, rate, bits int, data []byte) []byte {
	return append(wavFormat{1, channels, rate, bits}.header(uint32(len(data))), data...)
}

// TestParseWAV reads a stereo 16-bit file as mono and an unsigned 8-bit one, and refuses files that are not WAV,
// hold no channels, an unread sample format or rate, or a float sample that is not a number.
func TestParseWAV(t *testing.T) {
	stereo := []byte{0, 0x40, 0, 0xc0, 0, 0x20, 0, 0x20}
	samples, rate, err := decodeWAV(wavFile(2, 48000, 16, stereo))
	if err != nil || rate != 48000 || len(samples) != 2 || samples[0] != 0 || samples[1] != 0.25 {
		t.Errorf("stereo 16-bit: %v %d %v", samples, rate, err)
	}

	samples, _, err = decodeWAV(wavFile(1, 8000, 8, []byte{0x80, 0xc0, 0}))
	if err != nil || len(samples) != 3 || samples[0] != 0 || samples[1] != 0.5 || samples[2] != -1 {
		t.Errorf("8-bit: %v %v", samples, err)
	}

	for name, data := range map[string][]byte{
		"not RIFF":      []byte("hello world, not a wav"),
		"short fmt":     []byte("RIFF\x00\x00\x00\x00WAVEfmt \x04\x00\x00\x00abcd"),
		"no channels":   wavFile(0, 24000, 16, []byte{0, 0}),
		"12-bit":        wavFile(1, 24000, 12, []byte{0, 0}),
		"odd rate":      wavFile(1, 44101, 16, []byte{0, 0}),
		"tiny rate":     wavFile(1, 1, 16, []byte{0, 0}),
		"many channels": wavFile(64, 24000, 16, make([]byte, 128)),
	} {
		if _, _, err := decodeWAV(data); err == nil {
			t.Errorf("%s: no error", name)
		}
	}

	notANumber := append(wavFormat{3, 1, 24000, 32}.header(4),
		binary.LittleEndian.AppendUint32(nil, math.Float32bits(float32(math.NaN())))...)
	if _, _, err := decodeWAV(notANumber); err == nil {
		t.Error("a float sample that is not a number was read")
	}
}

// TestWAVRoundTrip writes samples as a WAV file and reads them back within 16-bit rounding.
func TestWAVRoundTrip(t *testing.T) {
	audio := []float32{0, 0.5, -0.5, 0.25, -1, 1}
	samples, rate, err := decodeWAV(EncodeWAV(audio))
	if err != nil || rate != SampleRate || len(samples) != len(audio) {
		t.Fatalf("decodeWAV = %v, %d, %v", samples, rate, err)
	}

	for i := range audio {
		if !approx.Equal(float64(samples[i]), float64(audio[i]), 1.0/32768, 0) {
			t.Errorf("sample %d: %f, want %f", i, samples[i], audio[i])
		}
	}
}

// decodeWAV is a WAV file's mono samples and rate, read as a reference is.
func decodeWAV(data []byte) ([]float32, int, error) {
	w, err := ParseWAV(data)
	if err != nil {
		return nil, 0, err
	}

	samples, err := w.Mono()
	return samples, w.Rate(), err
}

// TestWAVSecondsBeforeDecoding knows a file's length from its header and data size, before any sample is read.
func TestWAVSecondsBeforeDecoding(t *testing.T) {
	w, err := ParseWAV(wavFile(2, 8000, 16, make([]byte, 4*8000*3)))
	if err != nil || w.Seconds() != 3 || w.Rate() != 8000 {
		t.Errorf("a 3 s stereo file: %v s at %d, %v", w.Seconds(), w.Rate(), err)
	}
}
