package engine

import (
	"context"
	"errors"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/tokenizer"
)

// TestReferenceCodes runs the whole reference path: speech trim, whole frames, encoding, the end frame dropped and
// whole steps kept.
func TestReferenceCodes(t *testing.T) {
	e, g := releaseEngine(t)
	wav := fixtures.Floats(t, g.Reference.WAV24)
	reference := Reference{samples: wav, rate: audio.SampleRate, text: g.Prompts.ReferenceText}
	codes, err := e.referenceCodes(t.Context(), reference)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(codes, g.Reference.Codes) {
		t.Errorf("reference codes differ: %d codes, want %d", len(codes), len(g.Reference.Codes))
	}
}

// TestReferenceFramesKeepWholeSteps checks that a reference drops DroppedEndFrames frames and keeps whole steps, for an
// odd and an even number of frames, a part frame after them aside.
func TestReferenceFramesKeepWholeSteps(t *testing.T) {
	for _, c := range []struct{ frames, dropped, kept int }{
		{35, 1, 34}, {36, 1, 34}, {35, 2, 32}, {36, 2, 34}, {37, 0, 36}, {1, 2, 0},
	} {
		s := settings{DroppedEndFrames: c.dropped}
		if got := s.referenceFrames(c.frames*audio.FrameSamples + audio.FrameSamples/2); got != c.kept {
			t.Errorf("%d frames dropping %d: %d kept, want %d", c.frames, c.dropped, got, c.kept)
		}
	}
}

// TestReferenceCut cuts many clips: each starts ClipLeadSeconds before its speech and holds whole frames whose last
// one, dropped, leaves at most ClipTailSeconds after the speech and at most a frame less.
func TestReferenceCut(t *testing.T) {
	s := settings{ClipLeadSeconds: 0.1, ClipTailSeconds: 0.17}
	frame := float64(audio.FrameSamples) / audio.SampleRate
	rng := rand.New(rand.NewPCG(1, 2))
	for range 500 {
		speechStart := 0.2 + 0.8*rng.Float64()
		speechEnd := speechStart + 2 + 6*rng.Float64()
		samples := int((speechEnd + 0.4 + 0.6*rng.Float64()) * audio.SampleRate)
		start, end := s.referenceCut(speechStart, speechEnd, samples)
		if (end-start)%audio.FrameSamples != 0 || start != int((speechStart-s.ClipLeadSeconds)*audio.SampleRate) {
			t.Fatalf("cut %d to %d for speech from %.3f s", start, end, speechStart)
		}

		// Training drops a clip's last frame, leaving up to ClipTailSeconds after the speech and at most one frame
		// less.
		tail := float64(end-audio.FrameSamples)/audio.SampleRate - speechEnd
		if tail < s.ClipTailSeconds-frame || tail > s.ClipTailSeconds {
			t.Fatalf("tail %.4f s after speech ending at %.3f s", tail, speechEnd)
		}
	}

	if start, end := s.referenceCut(0.5, 3, int(3.1*audio.SampleRate)); end > int(3.1*audio.SampleRate) ||
		(end-start)%audio.FrameSamples != 0 {
		t.Errorf("a recording that ends soon was cut from %d to %d", start, end)
	}
}

// TestReferenceSpeechKeepsATaggedEdge keeps the clip's edge on a side whose text holds a sound tag, and cuts to the
// speech elsewhere.
func TestReferenceSpeechKeepsATaggedEdge(t *testing.T) {
	samples := 9 * audio.SampleRate
	for referenceText, want := range map[string][2]float64{
		"For an evening out.":         {1.2, 7.9},
		"<laugh> For an evening out.": {0, 7.9},
		"For an evening out. <sigh>":  {1.2, 9},
		"For a <laugh> evening out.":  {1.2, 7.9},
	} {
		if first, last := referenceSpeech(referenceText, 1.2, 7.9, samples); [2]float64{first, last} != want {
			t.Errorf("%q: speech %.1f to %.1f s, want %v", referenceText, first, last, want)
		}
	}
}

// TestPrompts builds a voice's prompts from written texts of one chunk each and checks them, and each text's units,
// against the rows training lays out.
func TestPrompts(t *testing.T) {
	e, g := promptEngine(t), fixtures.LoadGolden(t)
	static, err := voicePrompt(e, g.Prompts.ReferenceText, g.Reference.Codes)
	if err != nil {
		t.Fatal(err)
	}

	if len(g.Prompts.Texts) == 0 {
		t.Fatal("the golden file has no prompt cases")
	}

	for i, written := range g.Prompts.Texts {
		chunks := chunksOf(t, e, written)
		if len(chunks) != 1 {
			t.Fatalf("%q is %d chunks, want one", written, len(chunks))
		}

		open, _ := chunks[0].split(e.settings.LookaheadUnits)
		prompt := e.markers.appendOpening(e.markers.appendChunkStatic(slices.Clone(static), chunks[0]), open)
		sameUnits := slices.EqualFunc(chunks[0].units, g.Prompts.Units[i], slices.Equal)
		if !slices.Equal(prompt, g.Prompts.IDs[i]) || !sameUnits {
			t.Errorf("%q differs: %d prompt tokens and %d units, want %d and %d", written, len(prompt),
				len(chunks[0].units), len(g.Prompts.IDs[i]), len(g.Prompts.Units[i]))
		}
	}
}

// TestReferenceLengthLimits refuses a clip whose speech is shorter or longer than any training reference's.
func TestReferenceLengthLimits(t *testing.T) {
	e, g := releaseEngine(t)
	wav := fixtures.Floats(t, g.Reference.WAV24)
	if len(wav) < 6*audio.SampleRate {
		t.Fatalf("the golden clip is %d samples, too short to repeat into 12 s", len(wav))
	}

	for _, seconds := range []float64{1, 12} {
		clip := slices.Repeat(wav, 2)[:int(seconds*audio.SampleRate)]
		_, err := e.referenceCodes(t.Context(), Reference{samples: clip, rate: audio.SampleRate, text: ""})
		if !errors.Is(err, ErrBadReference) || !strings.Contains(err.Error(), "of speech") {
			t.Errorf("a %.0f s reference: %v, want a refusal of its length of speech", seconds, err)
		}
	}
}

// TestReferencesWithoutSpeechAreRefused refuses a clip of silence or of noise, which holds no speech, and a text that
// holds nothing to say: training references were all spoken words.
func TestReferencesWithoutSpeechAreRefused(t *testing.T) {
	e, g := releaseEngine(t)
	rng := rand.New(rand.NewPCG(3, 4))
	silence, noise := make([]float32, 5*audio.SampleRate), make([]float32, 5*audio.SampleRate)
	for i := range noise {
		noise[i] = float32(rng.NormFloat64() * 0.1)
	}

	for name, clip := range map[string][]float32{"silence": silence, "noise": noise} {
		reference := Reference{samples: clip, rate: audio.SampleRate, text: g.Prompts.ReferenceText}
		_, err := e.referenceCodes(t.Context(), reference)
		if !errors.Is(err, ErrBadReference) {
			t.Errorf("a clip of %s: %v, want a bad reference", name, err)
		}
	}

	for _, referenceText := range []string{"", "   ", "<br>", "<laugh>"} {
		if _, err := e.NewReference(nil, referenceText); !errors.Is(err, ErrBadReference) {
			t.Errorf("reference text %q: %v, want a bad reference", referenceText, err)
		}
	}
}

// TestParseSettings reads a release's settings, refuses each one missing or out of range, and takes overrides only of
// settings the engine reads.
func TestParseSettings(t *testing.T) {
	good := `{"temperature": 0.7, "guidance_scale": 2, "max_seconds": 40, "chunk_tokens": 50, "chunk_sentences": 2,
		"sentence_tokens": 100, "lookahead_units": 2, "main_window": 32, "head_past_steps": 2, "sample_rate": 24000,
		"frame_samples": 2048, "frame_tokens": 7, "slot_levels": [0, 1, 2, 2, 1, 2, 2], "level_size": 4096,
		"step_frames": 2,
		"clip_lead_seconds": 0.1, "clip_tail_seconds": 0.17, "dropped_end_frames": 1, "reference_ms": [1500, 10000],
		"score_rate": 16000, "vad_threshold": 0.5, "vad_negative_offset": 0.15, "vad_min_speech_ms": 250,
		"vad_min_silence_ms": 100, "vad_speech_pad_ms": 30}`
	_, err := parseSettings([]byte(good), nil)
	if err != nil {
		t.Fatal(err)
	}

	for name, data := range map[string]string{
		"no sentences":   strings.Replace(good, `"chunk_sentences": 2`, `"chunk_sentences": 0`, 1),
		"missing":        strings.Replace(good, `"chunk_tokens": 50,`, ``, 1),
		"missing VAD":    strings.Replace(good, `"vad_min_speech_ms": 250,`, ``, 1),
		"empty":          `{}`,
		"no look-ahead":  strings.Replace(good, `"lookahead_units": 2`, `"lookahead_units": 0`, 1),
		"no window":      strings.Replace(good, `"main_window": 32`, `"main_window": 0`, 1),
		"negative tail":  strings.Replace(good, `"clip_tail_seconds": 0.17`, `"clip_tail_seconds": -0.17`, 1),
		"reversed range": strings.Replace(good, `[1500, 10000]`, `[10000, 1500]`, 1),
		"short frame":    strings.Replace(good, `[0, 1, 2, 2, 1, 2, 2]`, `[0, 1, 2, 2, 1, 2, 1]`, 1),
		"no such level":  strings.Replace(good, `[0, 1, 2, 2, 1, 2, 2]`, `[0, 1, 2, 2, 1, 2, 3]`, 1),
		"missing slot":   strings.Replace(good, `[0, 1, 2, 2, 1, 2, 2]`, `[0, 1, 2, 2, 1, 2]`, 1),
		"past steps":     strings.Replace(good, `"head_past_steps": 2`, `"head_past_steps": -1`, 1),
		"under a step":   strings.Replace(good, `"max_seconds": 40`, `"max_seconds": 0.1`, 1),
	} {
		if _, err := parseSettings([]byte(data), nil); err == nil {
			t.Errorf("%s: settings were accepted", name)
		}
	}

	release, err := parseSettings([]byte(good), []byte(`{"temperature": 0.5, "main_window": 64}`))
	if err != nil || release.Temperature != 0.5 || release.MainWindow != 64 || release.ChunkTokens != 50 {
		t.Errorf("overridden settings: %+v, %v", release, err)
	}

	for name, overrides := range map[string]string{
		"misspelt":        `{"temprature": 0.5}`,
		"read by no code": `{"sample_rate": 16000}`,
		"out of range":    `{"temperature": 0}`,
		"not JSON":        `temperature=0.5`,
		"two JSON values": `{"temperature": 0.5} {}`,
	} {
		if _, err := parseSettings([]byte(good), []byte(overrides)); err == nil {
			t.Errorf("%s overrides were accepted", name)
		}
	}
}

var errClosed = errors.New("listener gone")

// TestSpeak checks that speech arrives in more than one span of whole frames, that one seed speaks the same audio
// twice, and that a failed emit ends the speech with its own error, receives nothing more and stops generation at
// once.
func TestSpeak(t *testing.T) {
	e, _ := releaseEngine(t)
	script := scriptOf(t, e, "Hello there. How are you? I am fine. Thank you.")

	var spans [][]float32
	_, err := e.Speak(t.Context(), e.noVoice, script, rand.NewPCG(1, 2), func(audio []float32) error {
		spans = append(spans, slices.Clone(audio))
		return nil
	})
	if err != nil || len(spans) < 2 {
		t.Fatalf("%d spans, error %v", len(spans), err)
	}

	for i, span := range spans {
		if len(span) == 0 || len(span)%audio.FrameSamples != 0 {
			t.Errorf("span %d holds %d samples", i, len(span))
		}
	}

	// The first span comes after firstSpanSteps steps, the frames the decoder's first span and its context hold.
	const firstSpanSteps = 8
	source := &countingSource{Source: rand.NewPCG(1, 2)}
	emits := 0
	_, err = e.Speak(t.Context(), e.noVoice, script, source, func(audio []float32) error {
		emits++
		if !slices.Equal(audio, spans[0]) {
			t.Error("one seed gave two different first spans")
		}
		return errClosed
	})
	if !errors.Is(err, errClosed) || emits != 1 {
		t.Errorf("after a failed emit: error %v, %d emits", err, emits)
	}

	if steps := source.steps(); steps != firstSpanSteps {
		t.Errorf("generation wrote %d steps, want it to stop after the %d of the first span", steps, firstSpanSteps)
	}

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	source = &countingSource{Source: rand.NewPCG(1, 2)}
	_, err = e.Speak(ctx, e.noVoice, script, source, func([]float32) error {
		cancel()
		return nil
	})
	if !errors.Is(err, context.Canceled) || source.steps() != firstSpanSteps {
		t.Errorf("after ctx ended: error %v, %d steps, want it to stop after %d", err, source.steps(), firstSpanSteps)
	}
}

// TestSpeakRefusesMissingArguments refuses a voice another engine made, a nil voice, source or emit, and NewVoice
// refuses a reference another engine made and stops when ctx has ended.
func TestSpeakRefusesMissingArguments(t *testing.T) {
	e, g := releaseEngine(t)
	script := scriptOf(t, e, "Hi.")
	emit := func([]float32) error { return nil }
	for name, call := range map[string]func() error{
		"foreign voice": func() error {
			_, err := e.Speak(t.Context(), &Voice{}, script, rand.NewPCG(1, 2), emit)
			return err
		},
		"nil voice": func() error {
			_, err := e.Speak(t.Context(), nil, script, rand.NewPCG(1, 2), emit)
			return err
		},
		"nil source": func() error {
			_, err := e.Speak(t.Context(), e.noVoice, script, nil, emit)
			return err
		},
		"nil emit": func() error {
			_, err := e.Speak(t.Context(), e.noVoice, script, rand.NewPCG(1, 2), nil)
			return err
		},
		"foreign reference": func() error {
			_, err := e.NewVoice(t.Context(), Reference{})
			return err
		},
	} {
		if call() == nil {
			t.Errorf("%s was accepted", name)
		}
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	reference := Reference{engine: e, samples: fixtures.Floats(t, g.Reference.WAV24), rate: audio.SampleRate,
		text: g.Prompts.ReferenceText}
	if _, err := e.NewVoice(ctx, reference); !errors.Is(err, context.Canceled) {
		t.Errorf("NewVoice after ctx ended: %v", err)
	}
}

// TestConcurrentSpeakSpeaksAsAlone speaks two scripts of several chunks alone, each from a seed of its own, then both
// at once: each must speak the audio it spoke alone. The two differ in text and seed, so a buffer or tail the requests
// shared would carry one request's speech into the other's.
func TestConcurrentSpeakSpeaksAsAlone(t *testing.T) {
	e, _ := releaseEngine(t)
	scripts := [2]Script{
		scriptOf(t, e, "Hello there. How are you? I am fine."),
		scriptOf(t, e, "The train was late. We waited. It rained."),
	}
	speak := func(i int) ([]float32, error) { return speakAll(t, e, scripts[i], rand.NewPCG(uint64(i), 6)) }

	var alone [2][]float32
	for i := range alone {
		var err error
		alone[i], err = speak(i)
		if err != nil {
			t.Fatal(err)
		}
	}

	var together [2][]float32
	var errs [2]error
	var wg sync.WaitGroup
	for i := range together {
		wg.Go(func() { together[i], errs[i] = speak(i) })
	}
	wg.Wait()
	for i := range together {
		if errs[i] != nil || !slices.Equal(together[i], alone[i]) {
			t.Errorf("request %d spoken beside another: error %v, %d samples, want the %d spoken alone", i, errs[i],
				len(together[i]), len(alone[i]))
		}
	}
}

// TestDecoderNoiseFollowsTheSeed speaks a script twice from sources that differ only in the two values Speak draws
// first, the decoder noise's seed: the codes are the same, so the audio is as long, and the noise differs, so the
// samples do.
func TestDecoderNoiseFollowsTheSeed(t *testing.T) {
	e, _ := releaseEngine(t)
	script := scriptOf(t, e, "Hello there.")
	var spoken [2][]float32
	for i := range spoken {
		var err error
		spoken[i], err = speakAll(t, e, script, &shiftedSource{Source: rand.NewPCG(1, 2), shift: uint64(i)})
		if err != nil {
			t.Fatal(err)
		}
	}

	if len(spoken[0]) != len(spoken[1]) || slices.Equal(spoken[0], spoken[1]) {
		t.Errorf("%d and %d samples, equal: %v; want as many, and different noise", len(spoken[0]), len(spoken[1]),
			slices.Equal(spoken[0], spoken[1]))
	}
}

// shiftedSource adds shift to the first speakDraws values of its source and passes the rest unchanged.
type shiftedSource struct {
	rand.Source
	shift uint64
	draws int
}

func (s *shiftedSource) Uint64() uint64 {
	s.draws++
	if s.draws <= speakDraws {
		return s.Source.Uint64() + s.shift
	}
	return s.Source.Uint64()
}

// speakDraws is how many values Speak takes from its source before the first step: the decoder noise's seed.
const speakDraws = 2

// countingSource counts the values generation draws, a step's codes taking one each.
type countingSource struct {
	rand.Source
	draws int
}

func (s *countingSource) Uint64() uint64 {
	s.draws++
	return s.Source.Uint64()
}

func (s *countingSource) steps() int {
	return (s.draws - speakDraws) / model.StepCodes
}

var (
	loadOnce    sync.Once
	testEngine  *Engine
	errTestLoad error
)

// promptEngine is an engine with the release file's tokenizer and settings and no models, read from the file's header
// alone: enough to build every prompt.
func promptEngine(tb testing.TB) *Engine {
	tb.Helper()
	meta := fixtures.LoadWeights(tb).Metadata
	release, err := parseSettings([]byte(meta["settings"]), nil)
	if err != nil {
		tb.Fatal(err)
	}

	textTokenizer, err := tokenizer.New([]byte(meta["tokenizer"]))
	if err != nil {
		tb.Fatal(err)
	}

	return newEngine(textTokenizer, release, 1)
}

// voicePrompt is the prompt NewVoice reads for a reference that says referenceText and encodes to codes.
func voicePrompt(e *Engine, referenceText string, codes []int) ([]int, error) {
	tokens, err := e.referenceTokens(referenceText)
	if err != nil {
		return nil, err
	}
	return e.markers.voicePrompt(tokens, codes), nil
}

// withSettings is the release engine with settings of its own, change applied, sharing the models, so a test that
// changes a setting changes it for no other test.
func withSettings(tb testing.TB, change func(*settings)) (*Engine, *fixtures.Golden) {
	tb.Helper()
	e, g := releaseEngine(tb)
	own := *e
	own.settings.SlotLevels = slices.Clone(e.settings.SlotLevels)
	change(&own.settings)
	// A request sized for other settings stays with the engine it served.
	own.requests = make(chan *request, 1)
	return &own, g
}

func releaseEngine(tb testing.TB) (*Engine, *fixtures.Golden) {
	tb.Helper()
	path, g := fixtures.WeightsPath(tb), fixtures.LoadGolden(tb)
	loadOnce.Do(func() {
		testEngine, errTestLoad = Load(path, Options{})
	})
	if errTestLoad != nil {
		tb.Fatal(errTestLoad)
	}
	return testEngine, g
}

// chunksOf is written text prepared and packed into chunks as Speak packs it when no chunk carries speech on.
func chunksOf(tb testing.TB, e *Engine, written string) []chunk {
	tb.Helper()
	chunks, err := e.packAll(scriptOf(tb, e, written).pieces)
	if err != nil {
		tb.Fatal(err)
	}

	return chunks
}

// scriptOf is written text prepared for Speak.
func scriptOf(tb testing.TB, e *Engine, written string) Script {
	tb.Helper()
	script, err := e.Prepare(written)
	if err != nil {
		tb.Fatal(err)
	}

	return script
}

// speakAll is the whole of script's speech in no voice, drawn from source.
func speakAll(tb testing.TB, e *Engine, script Script, source rand.Source) ([]float32, error) {
	tb.Helper()
	var out []float32
	_, err := e.Speak(tb.Context(), e.noVoice, script, source, func(samples []float32) error {
		out = append(out, samples...)
		return nil
	})
	return out, err
}

// stepSeconds is the MaxSeconds that caps a chunk at the given number of steps.
func stepSeconds(steps int) float64 {
	return float64(audio.FrameSamples*model.StepFrames) * float64(steps) / audio.SampleRate
}
