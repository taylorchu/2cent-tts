package audio

import (
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// TestVAD scores golden audio window by window and finds where its speech starts and ends as Silero VAD does.
func TestVAD(t *testing.T) {
	r, g := releaseCodecs(t), fixtures.LoadGolden(t)
	audio := fixtures.Floats(t, g.VAD.Audio)
	probabilities := r.vad.probabilities(audio)
	// The detector also scores a last, zero-padded window, which the golden values leave out.
	if len(probabilities) != len(g.VAD.Probabilities)+1 {
		t.Fatalf("%d windows, want the golden %d and the padded one", len(probabilities), len(g.VAD.Probabilities))
	}

	for i, want := range g.VAD.Probabilities {
		// The detector's LSTM carries a window's rounding into the next, so the bound is looser than one layer's.
		if !approx.Equal(float64(probabilities[i]), want, 1e-4, 0) {
			t.Errorf("window %d: probability %f, want %f", i, probabilities[i], want)
		}
	}

	spans := g.VAD.Spans
	start, end, ok := r.vad.SpeechBounds(audio, r.vadSettings)
	if !ok || len(spans) == 0 || start != spans[0][0] || end != spans[len(spans)-1][1] {
		t.Errorf("speech from %v to %v (%v), want the golden spans' edges %v", start, end, ok, spans)
	}
}
