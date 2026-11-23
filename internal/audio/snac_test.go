package audio

import (
	"math"
	"slices"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// TestSNACDecode decodes golden codes with golden noise and matches the training codec's audio.
func TestSNACDecode(t *testing.T) {
	r, g := releaseCodecs(t), fixtures.LoadGolden(t)
	audio := decodeWhole(r.snac, g.SNAC.DecodeCodes, func(position int) float32 {
		return float32(math.Sin(g.SNAC.NoiseRate*float64(position)) * g.SNAC.NoiseScale)
	})
	want := fixtures.Floats(t, g.SNAC.DecodeAudio)
	if len(audio) != len(want) {
		t.Fatalf("decoded %d samples, want %d", len(audio), len(want))
	}

	// Both decode with the release's bf16 kernels, but sum in other orders through many layers.
	if d := approx.MaxDifference(audio, want); d > 1e-5 {
		t.Errorf("decoded audio differs by up to %g", d)
	}
}

// TestSNACEncode encodes golden audio and matches the training codec's codes, laid out in the release's slot order.
func TestSNACEncode(t *testing.T) {
	r, g := releaseCodecs(t), fixtures.LoadGolden(t)
	codes := r.snac.Encode(fixtures.Floats(t, g.SNAC.EncodeAudio))
	levels := g.SNAC.EncodeCodes
	// Each frame takes its levels' next codes in the release's slot order.
	var want []int
	var next [LevelCount]int
	for range levels[0] {
		for _, level := range r.snac.slotLevels {
			want = append(want, levels[level][next[level]]+level*LevelSize)
			next[level]++
		}
	}

	if !slices.Equal(codes, want) {
		t.Errorf("encoded codes %v, want %v", codes, want)
	}
}

// BenchmarkSNACDecode decodes one second of audio.
func BenchmarkSNACDecode(b *testing.B) {
	r, g := releaseCodecs(b), fixtures.LoadGolden(b)

	codes := slices.Repeat(g.SNAC.DecodeCodes, 2)
	for b.Loop() {
		decodeWhole(r.snac, codes, func(int) float32 { return 0 })
	}
}

// TestStream decodes the reference's codes frame by frame, without noise, and must return the audio of one whole
// decode, sample for sample. Two spans and the context past them hold every join a longer speech has.
func TestStream(t *testing.T) {
	r, g := releaseCodecs(t), fixtures.LoadGolden(t)
	silent := func(int) float32 { return 0 }
	codes := g.Reference.Codes[:min(len(g.Reference.Codes), (2*decodeSpan+decodeContext+1)*FrameCodes)]
	decoder := r.snac.NewStream(silent)
	if !slices.Equal(append(addFrames(decoder, codes), decoder.Flush()...), decodeWhole(r.snac, codes, silent)) {
		t.Error("a chunk decoded in spans differs from the chunk decoded whole")
	}

	// The stream starts over after a chunk ends, and a chunk shorter than a span decodes when flushed.
	short := codes[:5*FrameCodes]
	if addFrames(decoder, short) != nil {
		t.Fatal("audio before a span was complete")
	}
	if !slices.Equal(decoder.Flush(), decodeWhole(r.snac, short, silent)) {
		t.Error("a short second chunk differs from the chunk decoded alone")
	}
}

// TestStreamAllocatesNothing decodes a chunk in a warm stream, which allocates nothing, and speech carried on across
// chunks, as a long script's is, which holds no more codes than a span and its context.
func TestStreamAllocatesNothing(t *testing.T) {
	r, g := releaseCodecs(t), fixtures.LoadGolden(t)
	// Two spans and the context past them: a span's audio is final, then the flush decodes the rest.
	codes := g.Reference.Codes[:min(len(g.Reference.Codes), (2*decodeSpan+decodeContext+1)*FrameCodes)]
	decoder := r.snac.NewStream(func(int) float32 { return 0 })
	// Unlike addFrames, add keeps none of the audio, which would allocate.
	add := func(codes []int) {
		for f := range len(codes) / FrameCodes {
			decoder.Add(codes[f*FrameCodes : (f+1)*FrameCodes])
		}
	}

	if n := fixtures.Allocations(1, func() { add(codes); decoder.Flush() }); n != 0 {
		t.Errorf("a chunk allocated %d times", n)
	}

	add(slices.Repeat(codes, 2))
	if held := len(decoder.codes) / FrameCodes; held > decodeSpan+2*decodeContext {
		t.Errorf("speech carried on holds %d frames, more than a span and its context", held)
	}
}

// decodeWhole decodes codes in one go, in memory of its own.
func decodeWhole(s *SNAC, codes []int, noise func(position int) float32) []float32 {
	return slices.Clone(s.decode(codes, &scratch{noise: noise}))
}

// addFrames adds codes to s frame by frame and returns the audio that became final.
func addFrames(s *Stream, codes []int) []float32 {
	var audio []float32
	for frame := range slices.Chunk(codes, FrameCodes) {
		audio = append(audio, s.Add(frame)...)
	}
	return audio
}
