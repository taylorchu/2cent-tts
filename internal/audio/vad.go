package audio

import (
	"fmt"
	"math"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// VAD is Silero VAD 16 kHz: a fixed STFT, four convolutions and an LSTM cell, one probability per window. It is only
// read once loaded, so any number of goroutines may share it.
type VAD struct {
	basis        []float32
	encoder      sequence
	inputWeight  []float32
	hiddenWeight []float32
	inputBias    []float32
	hiddenBias   []float32
	outWeight    []float32
	outBias      float32
	hidden       int
}

// VADSettings are silero_vad.get_speech_timestamps's settings. Every field must be set; the zero value finds speech
// everywhere.
type VADSettings struct {
	// Threshold is the probability at which speech starts.
	Threshold float64
	// NegativeOffset is how far below Threshold the probability falls before speech ends.
	NegativeOffset float64
	// MinSpeechMilliseconds is the shortest speech kept.
	MinSpeechMilliseconds int
	// MinSilenceMilliseconds is the shortest silence that ends speech.
	MinSilenceMilliseconds int
	// SpeechPadMilliseconds widens the speech on each side.
	SpeechPadMilliseconds int
}

const (
	vadWindow  = 512
	vadContext = 64
	vadFilter  = 256
	vadHop     = 128
)

// LoadVAD reads the speech detector's weights, named vad.*. A missing or misshapen tensor is the loader's error, which
// LoadVAD returns too.
func LoadVAD(l *tensor.Loader) (*VAD, error) {
	hidden := l.Dims("vad.decoder.rnn.weight_hh", 2)[1]
	// The detector is small, so a release keeps all of it in f32.
	v := &VAD{
		basis:        l.Float32s("vad.stft.forward_basis_buffer", vadFilter+2, 1, vadFilter),
		inputWeight:  l.Float32s("vad.decoder.rnn.weight_ih", 4*hidden, hidden),
		hiddenWeight: l.Float32s("vad.decoder.rnn.weight_hh", 4*hidden, hidden),
		inputBias:    l.Float32s("vad.decoder.rnn.bias_ih", 4*hidden),
		hiddenBias:   l.Float32s("vad.decoder.rnn.bias_hh", 4*hidden),
		outWeight:    l.Float32s("vad.decoder.decoder.2.weight", 1, hidden, 1),
		hidden:       hidden,
	}

	if b := l.Float32s("vad.decoder.decoder.2.bias", 1); b != nil {
		v.outBias = b[0]
	}

	for i, stride := range []int{1, 2, 2, 1} {
		encoder := loadConv(l, fmt.Sprintf("vad.encoder.%d.reparam_conv", i), stride, 1)
		v.encoder = append(v.encoder, encoder, relu{})
	}

	if l.Err() != nil {
		return nil, l.Err()
	}

	return v, nil
}

// probabilities is the speech probability of each window of vadWindow samples, the last one zero-padded.
func (v *VAD) probabilities(samples []float32) []float32 {
	sc := new(scratch)
	h, c := make([]float32, v.hidden), make([]float32, v.hidden)
	gates := make([]float32, 4*v.hidden)
	// A window's input: the context the window before left, the window, then its reflection padding.
	x := make([]float32, vadContext+vadWindow+vadContext)
	bins := vadFilter/2 + 1
	frames := (len(x)-vadFilter)/vadHop + 1
	var probabilities []float32
	for start := 0; start < len(samples); start += vadWindow {
		copy(x, x[vadWindow:vadWindow+vadContext])
		window := x[vadContext : vadContext+vadWindow]
		clear(window[copy(window, samples[start:min(start+vadWindow, len(samples))]):])

		// Reflection padding on the right, without repeating the edge sample.
		n := vadContext + vadWindow
		for j := range vadContext {
			x[n+j] = x[n-2-j]
		}

		spectrum := sc.signal(bins, frames)
		for f := range bins {
			re, im := v.basis[f*vadFilter:][:vadFilter], v.basis[(f+bins)*vadFilter:][:vadFilter]
			for t := range frames {
				segment := x[t*vadHop:][:vadFilter]
				a, b := tensor.Dot(re, segment), tensor.Dot(im, segment)
				spectrum.row(f)[t] = float32(math.Sqrt(float64(a*a + b*b)))
			}
		}

		encoded := v.encoder.apply(spectrum, sc)
		input := encoded.data
		for g := range gates {
			gates[g] = tensor.Dot(v.inputWeight[g*v.hidden:][:v.hidden], input) + v.inputBias[g] +
				tensor.Dot(v.hiddenWeight[g*v.hidden:][:v.hidden], h) + v.hiddenBias[g]
		}
		sc.release(encoded)
		sc.release(spectrum)

		for j := range v.hidden {
			in, forget := tensor.Sigmoid(gates[j]), tensor.Sigmoid(gates[v.hidden+j])
			cell, out := float32(math.Tanh(float64(gates[2*v.hidden+j]))), tensor.Sigmoid(gates[3*v.hidden+j])
			c[j] = forget*c[j] + in*cell
			h[j] = out * float32(math.Tanh(float64(c[j])))
		}

		logit := v.outBias
		for j, value := range h {
			logit += v.outWeight[j] * max(value, 0)
		}
		probabilities = append(probabilities, tensor.Sigmoid(logit))
	}
	return probabilities
}

// SpeechBounds is where the speech that silero_vad.get_speech_timestamps finds with settings starts and ends, for
// audio at VADRate, in seconds rounded to a tenth as the training data stores them. Only the first span's start and
// the last span's end are padded, as only they are returned. ok is false when the audio holds no speech.
func (v *VAD) SpeechBounds(samples []float32, settings VADSettings) (start, end float64, ok bool) {
	threshold, negative := settings.Threshold, settings.Threshold-settings.NegativeOffset
	minSpeech := VADRate * settings.MinSpeechMilliseconds / 1000
	minSilence := VADRate * settings.MinSilenceMilliseconds / 1000
	pad := VADRate * settings.SpeechPadMilliseconds / 1000
	first, last := -1, 0
	keep := func(spanStart, spanEnd int) {
		if spanEnd-spanStart <= minSpeech {
			return
		}

		if first < 0 {
			first = spanStart
		}
		last = spanEnd
	}

	triggered := false
	spanStart, silenceStart := 0, 0
	for i, p := range v.probabilities(samples) {
		at := i * vadWindow
		if float64(p) >= threshold && silenceStart != 0 {
			silenceStart = 0
		}

		if float64(p) >= threshold && !triggered {
			triggered, spanStart = true, at
			continue
		}

		if float64(p) < negative && triggered {
			if silenceStart == 0 {
				silenceStart = at
			}

			if at-silenceStart < minSilence {
				continue
			}

			keep(spanStart, silenceStart)
			triggered, silenceStart = false, 0
		}
	}

	if triggered {
		keep(spanStart, len(samples))
	}

	if first < 0 {
		return 0, 0, false
	}

	first, last = max(0, first-pad), min(len(samples), last+pad)
	seconds := float64(len(samples)) / VADRate
	return roundTenth(float64(first) / VADRate), min(roundTenth(float64(last)/VADRate), seconds), true
}

// roundTenth rounds seconds to a tenth, as the training data's speech times were.
func roundTenth(x float64) float64 {
	return math.Round(x*10) / 10
}
