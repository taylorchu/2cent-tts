package engine

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/text"
)

// settings are the inference choices and the model and codec layout the release file holds, so the engine speaks
// with the values the model was trained and measured with instead of copies of them.
type settings struct {
	Temperature    float64 `json:"temperature"`
	GuidanceScale  float64 `json:"guidance_scale"`
	MaxSeconds     float64 `json:"max_seconds"`
	ChunkTokens    int     `json:"chunk_tokens"`
	ChunkSentences int     `json:"chunk_sentences"`
	SentenceTokens int     `json:"sentence_tokens"`
	// The units of a text held ahead of the speech, the positions behind it the main model attends to, and the
	// steps the head remembers, as the model was trained.
	LookaheadUnits int `json:"lookahead_units"`
	MainWindow     int `json:"main_window"`
	HeadPastSteps  int `json:"head_past_steps"`
	// The SNAC level of each slot of a frame, in the order the model writes them. The codec's other constants are the
	// engine's own: a release made for another codec has other tensor shapes, and Load refuses those.
	SlotLevels []int `json:"slot_levels"`
	// How a reference is cut, as training clips were, and how long its speech may run, in milliseconds.
	ClipLeadSeconds       float64    `json:"clip_lead_seconds"`
	ClipTailSeconds       float64    `json:"clip_tail_seconds"`
	DroppedEndFrames      int        `json:"dropped_end_frames"`
	ReferenceMilliseconds [2]float64 `json:"reference_ms"`
	// The speech detector's span rules, as audio.VADSettings holds them.
	VADThreshold              float64 `json:"vad_threshold"`
	VADNegativeOffset         float64 `json:"vad_negative_offset"`
	VADMinSpeechMilliseconds  int     `json:"vad_min_speech_ms"`
	VADMinSilenceMilliseconds int     `json:"vad_min_silence_ms"`
	VADSpeechPadMilliseconds  int     `json:"vad_speech_pad_ms"`
}

func (s settings) vad() audio.VADSettings {
	return audio.VADSettings{Threshold: s.VADThreshold, NegativeOffset: s.VADNegativeOffset,
		MinSpeechMilliseconds: s.VADMinSpeechMilliseconds, MinSilenceMilliseconds: s.VADMinSilenceMilliseconds,
		SpeechPadMilliseconds: s.VADSpeechPadMilliseconds}
}

// maxSteps is how many steps one chunk may run to: MaxSeconds of whole steps.
func (s settings) maxSteps() int {
	return int(s.MaxSeconds * audio.SampleRate / audio.FrameSamples / model.StepFrames)
}

func (s settings) limits() text.Limits {
	return text.Limits{Tokens: s.ChunkTokens, Sentences: s.ChunkSentences, SentenceTokens: s.SentenceTokens}
}

// parseSettings reads the release's settings, then the overrides over them. An override that names no setting is
// refused, as a misspelt one would otherwise change nothing unseen.
func parseSettings(data, overrides []byte) (settings, error) {
	var release map[string]json.RawMessage
	err := json.Unmarshal(data, &release)
	if err != nil {
		return settings{}, err
	}

	err = checkSettingsPresent(release)
	if err != nil {
		return settings{}, err
	}

	var s settings
	err = json.Unmarshal(data, &s)
	if err != nil {
		return settings{}, err
	}

	if len(overrides) > 0 {
		if !json.Valid(overrides) {
			return settings{}, errors.New("overrides are not one JSON value")
		}

		decoder := json.NewDecoder(bytes.NewReader(overrides))
		decoder.DisallowUnknownFields()
		err = decoder.Decode(&s)
		if err != nil {
			return settings{}, fmt.Errorf("overrides: %w", err)
		}
	}
	return s, s.check()
}

// checkSettingsPresent refuses a release file that leaves out a setting: a missing key would decode as zero, a value
// no release means.
func checkSettingsPresent(release map[string]json.RawMessage) error {
	for _, field := range reflect.VisibleFields(reflect.TypeFor[settings]()) {
		if key := field.Tag.Get("json"); release[key] == nil {
			return fmt.Errorf("%s is missing", key)
		}
	}
	return nil
}

// check refuses a setting out of range.
func (s settings) check() error {
	for _, rule := range []struct {
		setting string
		refused bool
	}{
		{"temperature", s.Temperature <= 0},
		{"guidance_scale", s.GuidanceScale <= 0},
		{"max_seconds", s.maxSteps() < 1},
		{"chunk_tokens", s.ChunkTokens <= 0},
		{"chunk_sentences", s.ChunkSentences <= 0},
		{"sentence_tokens", s.SentenceTokens < s.ChunkTokens},
		{"lookahead_units", s.LookaheadUnits <= 0},
		{"main_window", s.MainWindow <= 0},
		{"head_past_steps", s.HeadPastSteps < 0},
		{"slot_levels", audio.CheckSlotLevels(s.SlotLevels) != nil},
		{"clip_lead_seconds", s.ClipLeadSeconds < 0},
		{"clip_tail_seconds", s.ClipTailSeconds < 0},
		{"dropped_end_frames", s.DroppedEndFrames < 0},
		{"reference_ms", s.ReferenceMilliseconds[0] <= 0 || s.ReferenceMilliseconds[1] < s.ReferenceMilliseconds[0]},
		{"vad_threshold", s.VADThreshold <= s.VADNegativeOffset || s.VADThreshold >= 1},
		{"vad_min_speech_ms", s.VADMinSpeechMilliseconds < 0},
		{"vad_min_silence_ms", s.VADMinSilenceMilliseconds < 0},
		{"vad_speech_pad_ms", s.VADSpeechPadMilliseconds < 0},
	} {
		if rule.refused {
			return fmt.Errorf("%s is out of range", rule.setting)
		}
	}
	return nil
}
