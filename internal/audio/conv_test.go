package audio

import (
	"math/rand/v2"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// TestDense checks sizes that leave positions past a multiple of eight and rows past a multiple of four, with weights
// held in f32 and in bf16.
func TestDense(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, size := range [][3]int{{5, 7, 19}, {3, 8, 70}, {6, 66, 64}} {
		in, out, positions := size[0], size[1], size[2]
		x := newSignal(in, positions)
		w, b := make([]float32, out*in), make([]float32, out)
		for _, values := range [][]float32{x.data, w, b} {
			for i := range values {
				values[i] = float32(rng.NormFloat64())
			}
		}

		for _, weight := range []tensor.Values{fixtures.Values(t, tensor.F32, w), fixtures.Values(t, tensor.BF16, w)} {
			y := newSignal(out, positions)
			conv{in: in, out: out, kernel: 1, stride: 1, dilation: 1, weight: weight, bias: b}.dense(y, x,
				new(scratch))
			checkDense(t, size, y, x, weight.Float32(make([]float32, len(w))), b)
		}
	}
}

// checkDense compares y with the convolution of x by w, as f32, and b, summed in float64.
func checkDense(t *testing.T, size [3]int, y, x signal, w, b []float32) {
	t.Helper()
	in, out, positions := size[0], size[1], size[2]
	for o := range out {
		for s := range positions {
			want := float64(b[o])
			for i := range in {
				want += float64(w[o*in+i]) * float64(x.row(i)[s])
			}

			// The float32 kernel sums in tiles, against a float64 sum.
			if got := float64(y.row(o)[s]); !approx.Equal(got, want, 1e-4, 1e-4) {
				t.Fatalf("%v: output %d position %d = %f, want %f", size, o, s, got, want)
			}
		}
	}
}
