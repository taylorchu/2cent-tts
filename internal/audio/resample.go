package audio

import (
	"math"
	"slices"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// Resample is torchaudio.functional.resample with its defaults: a Hann-windowed sinc of its default width. Both rates
// are in samples per second and must be positive.
func Resample(x []float32, from, to int) []float32 {
	if from == to {
		return slices.Clone(x)
	}

	g := gcd(from, to)
	from, to = from/g, to/g
	const zeroCrossings, rolloff = 6.0, 0.99
	base := float64(min(from, to)) * rolloff
	width := int(math.Ceil(zeroCrossings * float64(from) / base))
	taps := 2*width + from
	kernel := make([]float32, to*taps)
	for j := range to {
		for m := range taps {
			t := (-float64(j)/float64(to) + float64(m-width)/float64(from)) * base
			t = max(-zeroCrossings, min(zeroCrossings, t))
			window := math.Pow(math.Cos(t*math.Pi/zeroCrossings/2), 2)
			t *= math.Pi
			value := 1.0
			if t != 0 {
				value = math.Sin(t) / t
			}
			kernel[j*taps+m] = float32(value * window * base / float64(from))
		}
	}

	padded := make([]float32, width+len(x)+width+from)
	copy(padded[width:], x)
	frames := (len(padded)-taps)/from + 1
	out := make([]float32, 0, frames*to)
	for f := range frames {
		for j := range to {
			out = append(out, tensor.Dot(kernel[j*taps:][:taps], padded[f*from:][:taps]))
		}
	}
	return out[:min(len(out), int(math.Ceil(float64(to)*float64(len(x))/float64(from))))]
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
