package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/text"
)

// Voice is a voice's prompt already read by the main model. Speak gives each chunk a copy of the voice's cache, so
// the reference is read once. A Voice works only with the Engine that made it, and it never changes, so any number
// of Speak calls may share it.
type Voice struct {
	engine *Engine
	prompt *model.Cache
}

// readVoice reads a voice's prompt in st.
func (e *Engine) readVoice(st *model.State, prompt []int) *Voice {
	cache := e.model.NewMainCache()
	// Each position of the voice's prompt attends to all before it, as in training, so it is read in one call.
	e.model.ReadMain(st, cache, nil, e.model.Inputs(st, prompt))
	return &Voice{engine: e, prompt: cache}
}

// Reference is a reference clip and what it says, checked as far as it can be before it is encoded: a WAV clip not
// too long, and a text that says something. Its samples are mono, at the clip's own rate. A Reference works only with
// the Engine that made it.
type Reference struct {
	engine  *Engine
	samples []float32
	rate    int
	// text is the written text, whose sound tags say where the clip's sound reaches its edges; tokens is its spoken
	// form, as the voice's prompt holds it.
	text   string
	tokens []int
}

// MaxReferenceTextCharacters is the longest reference text NewReference takes. A reference holds at most a few
// seconds of speech, so a longer text cannot be what the clip says.
const MaxReferenceTextCharacters = 1000

// NewReference checks a WAV clip that says referenceText, so a reference that can never be a voice is refused before
// any work is spent on it. The clip is 8- to 32-bit integer or float PCM at a common rate, any number of channels
// averaged to one, and at most a few times as long as the release's longest reference speech, read from its header.
// Tags that name no sound the model knows are dropped from referenceText, as from the text a voice speaks. Every
// refusal wraps ErrBadReference.
func (e *Engine) NewReference(clip []byte, referenceText string) (Reference, error) {
	referenceText = strings.TrimSpace(e.tokenizer.DropUnknownTags(referenceText))
	if n := utf8.RuneCountInString(referenceText); n > MaxReferenceTextCharacters {
		return Reference{}, fmt.Errorf("%w: the reference text is %d characters long, more than %d", ErrBadReference,
			n, MaxReferenceTextCharacters)
	}

	tokens, err := e.referenceTokens(referenceText)
	if err != nil {
		return Reference{}, err
	}

	wav, err := audio.ParseWAV(clip)
	if err != nil {
		return Reference{}, fmt.Errorf("%w: %w", ErrBadReference, err)
	}

	// Trimming takes off little more than leading and trailing quiet, so a much longer clip is refused from its header,
	// before any work is spent on its samples.
	limits := e.settings.ReferenceMilliseconds
	if seconds := wav.Seconds(); seconds > maxClipToSpeech*limits[1]/1000 {
		return Reference{}, fmt.Errorf("%w: the reference is %.2f s long; it needs %.1f to %.1f s of speech",
			ErrBadReference, seconds, limits[0]/1000, limits[1]/1000)
	}

	samples, err := wav.Mono()
	if err != nil {
		return Reference{}, fmt.Errorf("%w: %w", ErrBadReference, err)
	}
	return Reference{engine: e, samples: samples, rate: wav.Rate(), text: referenceText, tokens: tokens}, nil
}

// NewVoice cuts a checked reference to its speech, encodes it and reads the voice's prompt of its text and codes. A
// clip that holds no speech, or whose speech is shorter or longer than any training reference's, is refused with
// ErrBadReference; ctx ending while NewVoice waits for its turns gives ctx's error. An encode holds about a hundred
// megabytes, and at most ParallelEncodes run at once.
func (e *Engine) NewVoice(ctx context.Context, ref Reference) (*Voice, error) {
	if ref.engine != e {
		return nil, errors.New("the reference was not made by this engine's NewReference")
	}

	codes, err := e.referenceCodes(ctx, ref)
	if err != nil {
		return nil, err
	}

	prompt := e.markers.voicePrompt(ref.tokens, codes)
	// A voice reads its prompt once, so its state is not kept.
	var v *Voice
	err = e.withTurn(ctx, func() { v = e.readVoice(new(model.State), prompt) })
	return v, err
}

// maxClipToSpeech is how many times longer than the longest reference's speech a clip may be before trimming.
const maxClipToSpeech = 3

// referenceTokens is a reference's text as its voice's prompt holds it. Training took a reference's text as the clip's
// words between two cuts, often mid-sentence, so it is not closed with a period as a chunk's text is.
func (e *Engine) referenceTokens(referenceText string) ([]int, error) {
	// Every training reference was spoken words, so a text with none says nothing of how the clip's words sound.
	if !text.HasWord(referenceText) {
		return nil, fmt.Errorf("%w: the reference text holds no word", ErrBadReference)
	}

	spoken, err := text.Spoken(referenceText)
	if err != nil {
		return nil, fmt.Errorf("%w: reference text: %w", ErrBadReference, err)
	}

	c, err := e.encodeChunk(spoken)
	if err != nil {
		return nil, fmt.Errorf("%w: reference text: %w", ErrBadReference, err)
	}
	return c.textTokens(), nil
}

// referenceCodes cuts ref's clip the way training cut its references and encodes it in the model's frame layout.
func (e *Engine) referenceCodes(ctx context.Context, ref Reference) ([]int, error) {
	err := take(ctx, e.encodes)
	if err != nil {
		return nil, err
	}
	defer func() { <-e.encodes }()

	// The detector reads the clip at its own rate, resampled once, as training's did.
	speechStart, speechEnd, ok := e.vad.SpeechBounds(audio.Resample(ref.samples, ref.rate, audio.VADRate),
		e.settings.vad())
	// Every training reference holds speech; a clip of silence or noise would be a voice the model never heard.
	if !ok {
		return nil, fmt.Errorf("%w: the reference holds no speech", ErrBadReference)
	}

	clip := audio.Resample(ref.samples, ref.rate, audio.SampleRate)
	first, last := referenceSpeech(ref.text, speechStart, speechEnd, len(clip))
	start, end := e.settings.referenceCut(first, last, len(clip))
	clip = clip[start:end]
	frames := e.settings.referenceFrames(len(clip))
	milliseconds := float64(frames*audio.FrameSamples) * 1000 / audio.SampleRate
	if limits := e.settings.ReferenceMilliseconds; milliseconds < limits[0] || milliseconds > limits[1] {
		return nil, fmt.Errorf("%w: the reference cut to its speech is %.2f s long; it needs %.1f to %.1f s of speech",
			ErrBadReference, milliseconds/1000, limits[0]/1000, limits[1]/1000)
	}

	// The encoder reads past a frame's edges, so the dropped frames are encoded too, as training encoded a whole clip
	// before it dropped them, and the kept frames' codes are training's.
	return e.snac.Encode(clip)[:frames*audio.FrameCodes], nil
}

// referenceFrames is how many frames of a cut clip of sampleCount samples its reference keeps. Training dropped every
// clip's end frames, so no reference it saw ends on one, and read audio in whole steps.
func (s settings) referenceFrames(sampleCount int) int {
	frames := max(sampleCount/audio.FrameSamples-s.DroppedEndFrames, 0)
	return frames - frames%model.StepFrames
}

// referenceSpeech is where a reference's sound starts and ends, in seconds: its speech, except on a side whose text
// holds a sound tag, which keeps the clip's edge. The speech detector misses a laugh or a breath, while training cut
// clips by the times of words and tags alike, so a tag's sound stays in the clip with its text.
func referenceSpeech(referenceText string, speechStart, speechEnd float64, sampleCount int) (float64, float64) {
	opens, closes := text.TagEdges(referenceText)
	if opens {
		speechStart = 0
	}

	if closes {
		speechEnd = float64(sampleCount) / audio.SampleRate
	}
	return speechStart, speechEnd
}

// referenceCut is where a reference is cut the way a training clip was: from ClipLeadSeconds before the speech to the
// first whole frame at least ClipTailSeconds after it. A recording that ends sooner is cut at its last whole frame,
// since a user's clip has no room tone to fill with. referenceFrames then drops end frames, as training did.
func (s settings) referenceCut(speechStart, speechEnd float64, sampleCount int) (int, int) {
	start := max(0, int((speechStart-s.ClipLeadSeconds)*audio.SampleRate))
	frames := int(math.Ceil(((speechEnd+s.ClipTailSeconds)*audio.SampleRate - float64(start)) / audio.FrameSamples))
	frames = max(min(frames, (sampleCount-start)/audio.FrameSamples), 0)
	return start, start + frames*audio.FrameSamples
}
