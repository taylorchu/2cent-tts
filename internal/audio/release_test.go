package audio

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/fixtures"
	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// releaseSettings are the release settings the codec and the speech detector read.
type releaseSettings struct {
	SlotLevels             []int   `json:"slot_levels"`
	Threshold              float64 `json:"vad_threshold"`
	NegativeOffset         float64 `json:"vad_negative_offset"`
	MinSpeechMilliseconds  int     `json:"vad_min_speech_ms"`
	MinSilenceMilliseconds int     `json:"vad_min_silence_ms"`
	SpeechPadMilliseconds  int     `json:"vad_speech_pad_ms"`
}

type release struct {
	snac        *SNAC
	vad         *VAD
	vadSettings VADSettings
}

var (
	codecsOnce sync.Once
	codecs     release
	errCodecs  error
)

// releaseCodecs is the codec and the speech detector of the release weights file, loaded once.
func releaseCodecs(tb testing.TB) release {
	tb.Helper()
	w := fixtures.LoadWeights(tb)
	codecsOnce.Do(func() { codecs, errCodecs = loadCodecs(w) })
	if errCodecs != nil {
		tb.Fatal(errCodecs)
	}
	return codecs
}

func loadCodecs(w *tensor.Weights) (release, error) {
	var s releaseSettings
	err := json.Unmarshal([]byte(w.Metadata["settings"]), &s)
	if err != nil {
		return release{}, err
	}

	l := tensor.NewLoader(w)
	codec, err := LoadSNAC(l, s.SlotLevels)
	if err != nil {
		return release{}, err
	}

	detector, err := LoadVAD(l)
	if err != nil {
		return release{}, err
	}
	return release{snac: codec, vad: detector, vadSettings: VADSettings{s.Threshold, s.NegativeOffset,
		s.MinSpeechMilliseconds, s.MinSilenceMilliseconds, s.SpeechPadMilliseconds}}, nil
}

// TestLoadMissingTensors loads from a weights file with no tensors: every loader must report it, not panic.
func TestLoadMissingTensors(t *testing.T) {
	for name, load := range map[string]func(*tensor.Loader) error{
		"snac": func(l *tensor.Loader) error {
			_, err := LoadSNAC(l, []int{0, 1, 2, 2, 1, 2, 2})
			return err
		},
		"vad": func(l *tensor.Loader) error {
			_, err := LoadVAD(l)
			return err
		},
	} {
		if load(tensor.NewLoader(&tensor.Weights{})) == nil {
			t.Errorf("%s: loading from no tensors reported no error", name)
		}
	}
}
