package engine

import (
	"context"
	"errors"
	"math/rand/v2"

	"github.com/taylorchu/2cent-tts/internal/audio"
)

// Report is what Speak did.
type Report struct {
	// Capped is how many chunks reached the release's length cap before their speech ended; their speech stops there
	// and the next chunk starts afresh.
	Capped int
}

// Speak speaks a script in voice and hands the audio to emit as it is made, a span at a time, on the goroutine that
// called Speak. emit's samples are mono at audio.SampleRate and are the decoder's memory, read until emit returns.
// When emit fails, Speak stops and returns emit's error; it also stops when ctx ends. Speak draws from source, so calls
// that run at once each need a source of their own.
//
// A chunk that reaches the length cap stops there, and the rest of the script is still spoken. An empty script is
// refused with ErrBadText. The report counts the capped chunks spoken before any error.
func (e *Engine) Speak(ctx context.Context, voice *Voice, script Script, source rand.Source,
	emit func(samples []float32) error) (Report, error) {
	switch {
	case voice == nil || voice.engine != e:
		return Report{}, errors.New("the voice was not made by this engine")
	case source == nil || emit == nil:
		return Report{}, errors.New("Speak needs a source and an emit")
	case len(script.pieces) == 0:
		return Report{}, errNothingToSay
	}

	r := e.takeRequest()
	defer e.giveBack(r)
	r.seed(source)
	r.decoder.Reset()
	return r.speakChunks(ctx, voice, script, emit)
}

// speakChunks speaks the script a chunk at a time and decodes each step as it is written. The decoder is flushed after
// a chunk unless the next chunk carries its speech on; a chunk that does is packed after the text it carries, which
// counts against the chunk limits.
func (r *request) speakChunks(ctx context.Context, voice *Voice, script Script,
	emit func([]float32) error) (Report, error) {
	var report Report
	pending := script.pieces
	var carried *tail
	for len(pending) > 0 {
		next, rest, err := r.engine.nextChunk(pending, carried)
		if err != nil {
			return report, err
		}

		pending = rest
		err = r.generate(ctx, voice, next, carried, func(codes []int) error { return r.decode(codes, emit) })
		// Speech the cap cut off has no end to carry on from, so the next chunk starts afresh.
		carried = nil
		switch {
		case errors.Is(err, errCapped):
			report.Capped++
		case err != nil:
			return report, err
		case len(pending) > 0:
			r.tail.fill(&r.heard, next, r.engine.settings.MainWindow)
			carried = &r.tail
		}

		if carried == nil {
			err = emitSpan(r.decoder.Flush(), emit)
			if err != nil {
				return report, err
			}
		}
	}
	return report, nil
}

// decode adds a step's frames to the decoder and hands emit the audio that became final.
func (r *request) decode(codes []int, emit func([]float32) error) error {
	for at := 0; at < len(codes); at += audio.FrameCodes {
		err := emitSpan(r.decoder.Add(codes[at:at+audio.FrameCodes]), emit)
		if err != nil {
			return err
		}
	}
	return nil
}

func emitSpan(samples []float32, emit func([]float32) error) error {
	if len(samples) == 0 {
		return nil
	}
	return emit(samples)
}
