package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/tensor"
	"github.com/taylorchu/2cent-tts/internal/tokenizer"
)

// The contract between a release file and this engine; a new model keeps it and loads here.
const weightsFormat = "2cent-1"

// Engine speaks with the models of one release file. Nothing in it changes once Load returns but the memory it keeps
// for later requests, so it is safe for concurrent use.
type Engine struct {
	tokenizer *tokenizer.Tokenizer
	model     *model.Model
	snac      *audio.SNAC
	vad       *audio.VAD
	// noVoice is the voice's prompt with no reference clip.
	noVoice *Voice
	// unconditional is the static prompt of the pass guidance compares against, no reference and an empty text, read
	// once at load as a voice's prompt is.
	unconditional *model.Cache
	// unconditionalOpening is that pass's input where its target opens, the same for every chunk.
	unconditionalOpening [][]float32
	markers              markers
	settings             settings
	// turn gives one request's model call at a time every core; calls side by side would contend for the cores and
	// each run slower. It guards no memory: each request works in its own.
	turn chan struct{}
	// requests holds the memory of finished requests for later ones, up to Options.KeptRequests.
	requests chan *request
	// encodes bounds the reference encodes run at once: each holds the encoder's signals at the clip's full rate,
	// about a hundred megabytes for a long clip, outside any turn.
	encodes chan struct{}
}

// ParallelEncodes is how many references NewVoice encodes at once; later ones wait.
const ParallelEncodes = 2

// ParallelRequests is how many requests at once the worker pool serves on every core. Past it, a request whose parallel
// work finds every pool slot taken runs that work on its own core. The pool's calls are shared: one step runs at a
// time, other requests decode between their steps, and references encode beside both.
const ParallelRequests = tensor.MaxParallelCalls - ParallelEncodes - 1

// maxKeptRequests bounds Options.KeptRequests, far past what ParallelRequests serves well, so a mistyped count cannot
// reserve gigabytes.
const maxKeptRequests = 256

// ErrBadReference marks NewReference's and NewVoice's refusals of a reference clip or its text, which their caller can
// fix.
var ErrBadReference = errors.New("bad reference")

// Options are Load's choices.
type Options struct {
	// KeptRequests is how many finished requests' memory, about 20 MB each, the engine keeps for later Speak calls;
	// zero means one. It does not bound the Speak calls that run at once: a call past it makes memory of its own and
	// frees it after. More than ParallelRequests at once run part of their parallel work on one core.
	KeptRequests int
	// Settings, when not empty, is a JSON object of release settings that replace the file's, for measuring the model
	// with other choices, such as {"temperature": 0.5}. A name the file does not hold is refused.
	Settings json.RawMessage
}

// newEngine is an engine with no models yet.
func newEngine(textTokenizer *tokenizer.Tokenizer, release settings, requests int) *Engine {
	return &Engine{tokenizer: textTokenizer, settings: release, markers: markers(textTokenizer.Markers()),
		turn: make(chan struct{}, 1), requests: make(chan *request, max(requests, 1)),
		encodes: make(chan struct{}, ParallelEncodes)}
}

// withTurn runs work on a turn of the cores, once one is free, unless ctx ends first, and gives the turn back however
// work returns.
func (e *Engine) withTurn(ctx context.Context, work func()) error {
	err := take(ctx, e.turn)
	if err != nil {
		return err
	}
	defer func() { <-e.turn }()
	work()
	return nil
}

// take takes a place in slots once one is free, unless ctx ends first; the caller gives it back by receiving from
// slots.
func take(ctx context.Context, slots chan struct{}) error {
	// A select whose cases are both ready picks one at random, so an ended ctx is checked first.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	select {
	case slots <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Load reads a release file. The file is closed before Load returns, so an Engine holds none and needs no Close. An
// Engine shares the process's worker pool, so a process loads one.
func Load(path string, options Options) (*Engine, error) {
	if options.KeptRequests < 0 || options.KeptRequests > maxKeptRequests {
		return nil, fmt.Errorf("%d requests' memory to keep, want 0 to %d", options.KeptRequests, maxKeptRequests)
	}

	w, err := tensor.OpenWeights(path)
	if err != nil {
		return nil, err
	}
	// A file only read loses nothing when closing it fails.
	defer func() { _ = w.Close() }()

	if w.Metadata["format"] != weightsFormat {
		return nil, fmt.Errorf("%s is not a %s weights file", path, weightsFormat)
	}

	release, err := parseSettings([]byte(w.Metadata["settings"]), options.Settings)
	if err != nil {
		return nil, fmt.Errorf("settings: %w", err)
	}

	textTokenizer, err := tokenizer.New([]byte(w.Metadata["tokenizer"]))
	if err != nil {
		return nil, fmt.Errorf("tokenizer: %w", err)
	}

	if codes := textTokenizer.AudioCodeCount(); codes != audio.CodeCount {
		return nil, fmt.Errorf("the tokenizer holds %d audio codes, want %d", codes, audio.CodeCount)
	}

	e := newEngine(textTokenizer, release, options.KeptRequests)
	l := tensor.NewLoader(w)
	e.model, err = model.Load(l, []byte(w.Metadata["config"]),
		model.Layout{AudioBase: e.markers.AudioBase, SlotLevels: release.SlotLevels, PastSteps: release.HeadPastSteps})
	if err != nil {
		return nil, fmt.Errorf("model: %w", err)
	}

	e.snac, err = audio.LoadSNAC(l, release.SlotLevels)
	if err != nil {
		return nil, fmt.Errorf("snac: %w", err)
	}

	e.vad, err = audio.LoadVAD(l)
	if err != nil {
		return nil, fmt.Errorf("vad: %w", err)
	}

	// A release file and the engine that loads it must agree on every weight.
	if unread := l.Unread(); len(unread) > 0 {
		return nil, fmt.Errorf("weights: tensor %s is in the file but not in the engine's model", unread[0])
	}

	if e.model.TokenCount() < textTokenizer.TokenCount() {
		return nil, fmt.Errorf("the model reads %d token ids, fewer than the tokenizer's %d", e.model.TokenCount(),
			textTokenizer.TokenCount())
	}

	var st model.State
	e.noVoice = e.readVoice(&st, e.markers.noVoicePrompt())
	e.unconditional = e.readVoice(&st, e.markers.unconditionalPrompt()).prompt
	e.unconditionalOpening = e.model.Inputs(&st, e.markers.appendOpening(nil, nil))
	return e, nil
}

// NoVoice is the voice of no reference clip, in which the model speaks a voice of its own choosing. A library caller
// may speak in it; the command line asks for a voice.
func (e *Engine) NoVoice() *Voice {
	return e.noVoice
}
