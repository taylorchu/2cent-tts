// Package fixtures loads what the tests compare against: the release weights, the golden values PyTorch computed
// from them, and the normalizer corpus. A test skips when its file is missing, or fails when TWOCENT_REQUIRE_FIXTURES
// is 1. It also counts allocations for the tests that claim none. Only test files import it.
package fixtures

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// Golden holds the golden values, computed in PyTorch from the same weights file.
type Golden struct {
	Tokenizer []TokenizerCase            `json:"tokenizer"`
	Units     []UnitCase                 `json:"units"`
	Prompts   Prompts                    `json:"prompts"`
	Reference Reference                  `json:"reference"`
	Model     Generation                 `json:"model"`
	SNAC      Codec                      `json:"snac"`
	VAD       Detector                   `json:"vad"`
	Resample  map[string]json.RawMessage `json:"resample"`
}

// TokenizerCase is a text and the ids the tokenizer gives it.
type TokenizerCase struct {
	Text string `json:"text"`
	IDs  []int  `json:"ids"`
}

// UnitCase is a written text, its spoken form, its sentences and the tokens of each of its units, as training cut
// and encoded them.
type UnitCase struct {
	Text      string   `json:"text"`
	Spoken    string   `json:"spoken"`
	Sentences []string `json:"sentences"`
	Units     [][]int  `json:"units"`
}

// Prompts are written texts of one chunk each, the prompt training lays out for each in the reference's voice, and
// each text's units.
type Prompts struct {
	ReferenceText string    `json:"ref_text"`
	Texts         []string  `json:"texts"`
	IDs           [][]int   `json:"ids"`
	Units         [][][]int `json:"units"`
}

// Reference is a reference clip at 24 kHz and its codes.
type Reference struct {
	WAV24 string `json:"wav24"`
	Codes []int  `json:"codes"`
}

// Generation is what training reads in one pass over a row whose target is Codes and whose starts are Starts: the
// guided scores of each code's level and the decision's logit after each step.
type Generation struct {
	Prompt        []int     `json:"prompt"`
	Unconditional []int     `json:"unconditional"`
	Units         [][]int   `json:"units"`
	Codes         []int     `json:"codes"`
	Starts        []bool    `json:"starts"`
	Guided        []string  `json:"guided"`
	Decisions     []float64 `json:"decisions"`
}

// Codec is SNAC's decode of codes with a sine for its noise, and its encode of audio, one list of codes per level.
type Codec struct {
	DecodeCodes []int   `json:"decode_codes"`
	DecodeAudio string  `json:"decode_audio"`
	NoiseRate   float64 `json:"noise_rate"`
	NoiseScale  float64 `json:"noise_scale"`
	EncodeAudio string  `json:"encode_audio"`
	EncodeCodes [][]int `json:"encode_codes"`
}

// Detector is the speech detector's probability for each window of audio and the speech spans it finds.
type Detector struct {
	Audio         string       `json:"audio"`
	Probabilities []float64    `json:"probs"`
	Spans         [][2]float64 `json:"spans"`
}

type loaded[T any] struct {
	once   sync.Once
	value  T
	absent bool
	err    error
}

func (l *loaded[T]) get(tb testing.TB, path string, read func(path string) (T, error)) T {
	tb.Helper()
	l.once.Do(func() {
		_, err := os.Stat(path)
		if err != nil {
			l.absent = true
			return
		}
		l.value, l.err = read(path)
	})
	if l.absent {
		missing(tb, path)
	}

	if l.err != nil {
		tb.Fatal(l.err)
	}
	return l.value
}

var (
	weights         loaded[*tensor.Weights]
	golden          loaded[*Golden]
	normalizeCorpus loaded[[]byte]
)

// missing skips the test for want of the fixture at path, or fails it when TWOCENT_REQUIRE_FIXTURES is 1, so a run
// that must check everything cannot pass by skipping.
func missing(tb testing.TB, path string) {
	tb.Helper()
	if os.Getenv("TWOCENT_REQUIRE_FIXTURES") == "1" {
		tb.Fatal(path + " not found")
	}
	tb.Skip(path + " not found")
}

// weightsVariable names the release weights file, and weightsFile is its name under build/ when the variable is not
// set.
const weightsVariable, weightsFile = "TWOCENT_WEIGHTS", "2cent.safetensors"

// WeightsPath is the release weights file TWOCENT_WEIGHTS names, skipping the test without it.
func WeightsPath(tb testing.TB) string {
	tb.Helper()
	path := filePath(tb, weightsVariable, weightsFile)
	_, err := os.Stat(path)
	if err != nil {
		missing(tb, path)
	}
	return path
}

// LoadWeights is the release weights file TWOCENT_WEIGHTS names, opened once and kept open, skipping the test without
// it.
func LoadWeights(tb testing.TB) *tensor.Weights {
	tb.Helper()
	return weights.get(tb, filePath(tb, weightsVariable, weightsFile), tensor.OpenWeights)
}

// LoadGolden is the golden file TWOCENT_GOLDEN names, read once, skipping the test without it.
func LoadGolden(tb testing.TB) *Golden {
	tb.Helper()
	return golden.get(tb, filePath(tb, "TWOCENT_GOLDEN", "golden.json.gz"), readGolden)
}

// LoadNormalizeGolden is the gzipped normalizer corpus TWOCENT_NORMALIZE_GOLDEN names, read once, skipping the test
// without it.
func LoadNormalizeGolden(tb testing.TB) []byte {
	tb.Helper()
	return normalizeCorpus.get(tb, filePath(tb, "TWOCENT_NORMALIZE_GOLDEN", "normalize_golden.json.gz"), os.ReadFile)
}

func readGolden(path string) (*Golden, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	r, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	g := &Golden{}
	return g, json.NewDecoder(r).Decode(g)
}

func filePath(tb testing.TB, variable, name string) string {
	tb.Helper()
	if v := os.Getenv(variable); v != "" {
		return v
	}

	root, err := moduleRoot()
	if err != nil {
		tb.Fatal(err)
	}
	return filepath.Join(root, "build", name)
}

func moduleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		_, err := os.Stat(filepath.Join(dir, "go.mod"))
		if err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("no go.mod at or above the working directory")
		}
		dir = parent
	}
}

// Floats decodes base64 little-endian f32 values.
func Floats(tb testing.TB, encoded string) []float32 {
	tb.Helper()
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		tb.Fatal(err)
	}

	out := make([]float32, len(raw)/4)
	_, err = binary.Decode(raw, binary.LittleEndian, out)
	if err != nil {
		tb.Fatal(err)
	}
	return out
}

// Allocations is how many heap allocations the module's code makes in runs calls of f, after one call that sizes their
// memory. It leaves out what the runtime allocates for itself, such as the wait record of a goroutine that blocks, as
// the worker pool's do now and then: those come with the scheduler's timing, not with the code under test.
// Unlike testing.AllocsPerRun, which divides by the runs and rounds down, it counts every allocation of the module's.
func Allocations(runs int, f func()) int64 {
	// The first call sizes memory and is not counted, so profiling every allocation starts after it.
	f()
	old := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	defer func() { runtime.MemProfileRate = old }()
	before := moduleAllocations()
	for range runs {
		f()
	}
	return moduleAllocations() - before
}

// moduleAllocations is how many allocations the memory profile holds from the module's code, its own aside.
func moduleAllocations() int64 {
	// The profile counts an allocation once a collection after it has finished.
	runtime.GC()
	runtime.GC()
	n, _ := runtime.MemProfile(nil, true)
	records := make([]runtime.MemProfileRecord, n+64)
	n, _ = runtime.MemProfile(records, true)
	var total int64
	for _, record := range records[:n] {
		if madeByModule(record.Stack()) {
			total += record.AllocObjects
		}
	}
	return total
}

// madeByModule reports whether the module's code made an allocation of stack: not the runtime's wait record for a
// goroutine that blocks, and not this package.
func madeByModule(stack []uintptr) bool {
	frames := runtime.CallersFrames(stack)
	for {
		frame, more := frames.Next()
		switch {
		case frame.Function == "runtime.acquireSudog", strings.Contains(frame.Function, "/internal/fixtures."):
			return false
		case strings.HasPrefix(frame.Function, "github.com/taylorchu/2cent-tts/"):
			return true
		case !more:
			return false
		}
	}
}

// Values is values stored in a weights file as dtype and read back through a Loader, as a release's tensor is read.
func Values(tb testing.TB, dtype tensor.DType, values []float32) tensor.Values {
	tb.Helper()
	var body []byte
	if dtype == tensor.BF16 {
		body, _ = binary.Append(nil, binary.LittleEndian, tensor.AppendBF16(nil, values))
	} else {
		body, _ = binary.Append(nil, binary.LittleEndian, values)
	}

	header, err := json.Marshal(map[string]any{"x": map[string]any{"dtype": dtype, "shape": []int{len(values)},
		"data_offsets": []int{0, len(body)}}})
	if err != nil {
		tb.Fatal(err)
	}

	path := filepath.Join(tb.TempDir(), "values.safetensors")
	data := append(binary.LittleEndian.AppendUint64(nil, uint64(len(header))), header...)
	err = os.WriteFile(path, append(data, body...), 0o600)
	if err != nil {
		tb.Fatal(err)
	}

	w, err := tensor.OpenWeights(path)
	if err != nil {
		tb.Fatal(err)
	}
	defer func() { _ = w.Close() }()

	l := tensor.NewLoader(w)
	v := l.Values("x", dtype, len(values))
	if l.Err() != nil {
		tb.Fatal(l.Err())
	}
	return v
}
