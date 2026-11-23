package tensor

import (
	"fmt"
	"iter"
	"maps"
	"math"
	"math/rand/v2"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
)

// widths puts each vector width this CPU runs in use, the scalar kernels among them, and yields its name. The width
// in use before is in use again once the loop ends.
func widths() iter.Seq[string] {
	return func(yield func(string) bool) {
		saved := active
		defer func() { active = saved }()
		for _, lanes := range slices.Sorted(maps.Keys(kernelSets)) {
			active = kernelSets[lanes]
			if !yield(fmt.Sprintf("lanes=%d", lanes)) {
				return
			}
		}
	}
}

// TestCPUGetsItsKernels checks that the widths registered are the ones this CPU runs, that each width's set holds the
// kernels written for that width, and that the widest set is the one in use. A kernel of another width computes the
// right values, only slower, so no other test catches it.
func TestCPUGetsItsKernels(t *testing.T) {
	want := slices.Sorted(slices.Values(wantWidths()))
	if got := slices.Sorted(maps.Keys(kernelSets)); !slices.Equal(got, want) {
		t.Fatalf("widths %v registered, want %v", got, want)
	}

	for lanes, set := range kernelSets {
		suffix := "Scalar"
		if lanes > 0 {
			suffix = fmt.Sprintf("X%d", lanes)
		}

		for _, name := range kernelNames(set) {
			// The sixteen-lane set keeps the eight-lane packed kernel, since the convolution packs eight positions.
			if lanes == 16 && strings.HasSuffix(name, ".kernel4x8X8") {
				continue
			}

			if !strings.HasSuffix(name, suffix) {
				t.Errorf("the set at %d lanes holds %s", lanes, name)
			}
		}
	}

	if !slices.Equal(kernelNames(active), kernelNames(kernelSets[slices.Max(want)])) {
		t.Error("the set in use is not the widest")
	}
}

// kernelNames is the names of a set's functions.
func kernelNames(set kernelSet) []string {
	v := reflect.ValueOf(set)
	names := make([]string, v.NumField())
	for i := range names {
		names[i] = runtime.FuncForPC(v.Field(i).Pointer()).Name()
	}
	return names
}

// TestSnakeRow checks the snake activation at every width against its formula in float64.
func TestSnakeRow(t *testing.T) {
	for width := range widths() {
		t.Run(width, func(t *testing.T) {
			x := []float32{0, 0.5, -0.5, 1.7, -3, 10, -40, 123.25, 1e-4}
			for _, alpha := range []float32{0.05, 1, 3.7} {
				out := make([]float32, len(x))
				SnakeRow(out, x, alpha)
				for i, v := range x {
					sin := math.Sin(float64(alpha) * float64(v))
					want := float64(v) + sin*sin/float64(alpha)
					if !approx.Equal(float64(out[i]), want, 1e-5, 1e-5) {
						t.Errorf("SnakeRow alpha %g x %g: %g, want %g", alpha, v, out[i], want)
					}
				}
			}
		})
	}
}

// TestExpRow checks exp across the range the engine reaches, with the shifts and scales softmax and sampling use, and
// zero where the result would not be a normal float32.
func TestExpRow(t *testing.T) {
	for width := range widths() {
		t.Run(width, func(t *testing.T) {
			var x []float32
			for v := float32(-100); v <= 88; v += 0.37 {
				x = append(x, v)
			}
			x = append(x, -1000, float32(math.Inf(-1)), -87.3, 0, 1e-8, -1e-8, 88.3)
			for _, c := range [][2]float32{{1, 0}, {-1, 0}, {1 / 0.7, -3}, {0.125, 2.5}} {
				out := make([]float32, len(x))
				ExpRow(out, x, c[0], c[1])
				for i, v := range x {
					z := float64(v)*float64(c[0]) + float64(c[1])
					want := math.Exp(z)
					if z < expLow || z > expHigh {
						want = 0
						if z > expHigh {
							want = math.Exp(expHigh)
						}
					}

					// Rounding the argument to float32, fused or not, moves exp by up to its ulp in relative terms. An
					// infinite argument is clamped before any rounding, so only exp's own error is left.
					relative := 2e-6
					if !math.IsInf(z, 0) {
						relative += 0x1p-23 * math.Abs(z)
					}

					if !approx.Equal(float64(out[i]), want, 0, relative) {
						t.Errorf("ExpRow scale %g shift %g x %g: %g, want %g", c[0], c[1], v, out[i], want)
					}
				}
			}
		})
	}
}

// kernelInputs are one length's vector and four rows, with each row's plain dot product with the vector.
type kernelInputs struct {
	n        int
	x        []float32
	rows     [4][]float32
	rowsBF16 [4][]uint16
	want     [4]float64
	wantBF16 [4]float64
}

// TestKernels checks each width's kernels, the scalar ones among them, against plain loops, tails included.
func TestKernels(t *testing.T) {
	for width := range widths() {
		t.Run(width, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))
			for n := range 70 {
				in := newKernelInputs(rng, n)
				checkDots(t, in)
				start := randomFloats(rng, n)
				checkAxpy(t, in, start)
				checkAccumulateRows(t, rng, in, start)
				checkKernel4x8(t, in, randomFloats(rng, 8*n))
			}
		})
	}
}

// TestElementwiseKernelsIgnoreTheirSpan runs each kernel that works step by step over every span of a row. A step
// must come out the same wherever its span starts and ends, or audio decoded in spans differs from audio decoded
// whole. A scalar Axpy tail would fail this only on amd64, since arm64 fuses the scalar multiply-add too.
func TestElementwiseKernelsIgnoreTheirSpan(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	const n = 41
	x, start := randomFloats(rng, n), randomFloats(rng, n)
	for i := range x {
		x[i] *= 4
	}

	kernels := map[string]func(out, x, start []float32){
		"Axpy": func(out, x, start []float32) {
			copy(out, start)
			Axpy(out, 0.3, x)
		},
		"SnakeRow": func(out, x, _ []float32) { SnakeRow(out, x, 0.7) },
		"ExpRow":   func(out, x, _ []float32) { ExpRow(out, x, 1.3, -0.4) },
	}
	for width := range widths() {
		t.Run(width, func(t *testing.T) {
			for name, kernel := range kernels {
				if first, last := differentSpan(kernel, x, start); first >= 0 {
					t.Errorf("%s over steps %d to %d differs from the whole row", name, first, last)
				}
			}
		})
	}
}

// differentSpan is the first span whose steps the kernel computes otherwise than over the whole row, or -1.
func differentSpan(kernel func(out, x, start []float32), x, start []float32) (first, last int) {
	whole := make([]float32, len(x))
	kernel(whole, x, start)
	for first := range x {
		for last := first + 1; last <= len(x); last++ {
			span := make([]float32, last-first)
			kernel(span, x[first:last], start[first:last])
			if !slices.Equal(span, whole[first:last]) {
				return first, last
			}
		}
	}
	return -1, -1
}

func newKernelInputs(rng *rand.Rand, n int) kernelInputs {
	in := kernelInputs{n: n, x: randomFloats(rng, n)}
	for r := range in.rows {
		in.rows[r] = randomFloats(rng, n)
	}

	for r := range in.rows {
		in.rowsBF16[r] = toBF16(in.rows[r])
		in.want[r] = plainDot(in.x, func(i int) float32 { return in.rows[r][i] })
		in.wantBF16[r] = plainDot(in.x, func(i int) float32 { return bf16ToFloat32(in.rowsBF16[r][i]) })
	}
	return in
}

func checkDots(t *testing.T, in kernelInputs) {
	t.Helper()
	checkKernel(t, "Dot", in.n, 0, Dot(in.x, in.rows[0]), in.want[0])
	checkKernel(t, "DotBF16", in.n, 0, DotBF16(in.x, in.rowsBF16[0]), in.wantBF16[0])
	got := [4]float32{}
	got[0], got[1], got[2], got[3] = Dot4BF16(in.x, in.rowsBF16[0], in.rowsBF16[1], in.rowsBF16[2], in.rowsBF16[3])
	for r := range got {
		checkKernel(t, "Dot4BF16", in.n, r, got[r], in.wantBF16[r])
		// Attention scores keys in fours and the rest one at a time, so a key's score must not depend on which.
		if single := DotBF16(in.x, in.rowsBF16[r]); single != got[r] {
			t.Errorf("DotBF16 n=%d row %d: %v, Dot4BF16 gives %v", in.n, r, single, got[r])
		}
	}

	want := math.Inf(-1)
	var total float64
	for _, v := range in.x {
		want, total = max(want, float64(v)), total+float64(v)
	}
	if got := Max(in.x); float64(got) != want {
		t.Errorf("Max n=%d: %v, want %v", in.n, got, want)
	}
	checkKernel(t, "Sum", in.n, 0, Sum(in.x), total)

	// Pairing rows of a batch must change no result: each input of a pair matches Dot4BF16 run on it alone.
	w := in.rowsBF16
	first, second := dot4x2BF16(in.x, in.rows[0], w[0], w[1], w[2], w[3])
	for i, input := range [][]float32{in.x, in.rows[0]} {
		var want [4]float32
		want[0], want[1], want[2], want[3] = Dot4BF16(input, w[0], w[1], w[2], w[3])
		if got := [2][4]float32{first, second}[i]; got != want {
			t.Errorf("dot4x2BF16 n=%d input %d: %v, want %v", in.n, i, got, want)
		}
	}
}

func checkAxpy(t *testing.T, in kernelInputs, start []float32) {
	t.Helper()
	y := slices.Clone(start)
	Axpy(y, 0.3, in.x)
	for i := range y {
		checkKernel(t, "Axpy", in.n, i, y[i], float64(start[i])+0.3*float64(in.x[i]))
	}
}

func checkAccumulateRows(t *testing.T, rng *rand.Rand, in kernelInputs, start []float32) {
	t.Helper()
	n := in.n
	for _, count := range []int{0, 1, 5} {
		weights := randomFloats(rng, count)
		for _, stride := range []int{n, n + 3} {
			source := toBF16(randomFloats(rng, max(count-1, 0)*stride+n))
			out := slices.Clone(start)
			AccumulateRowsBF16(out, weights, source, stride)
			for i := range out {
				wantSum := float64(start[i])
				for p, weight := range weights {
					wantSum += float64(weight) * float64(bf16ToFloat32(source[p*stride+i]))
				}
				checkKernel(t, "AccumulateRowsBF16", n, i, out[i], wantSum)
			}
		}
	}
}

func checkKernel4x8(t *testing.T, in kernelInputs, packed []float32) {
	t.Helper()
	got := Kernel4x8(packed, in.rows[0], in.rows[1], in.rows[2], in.rows[3])
	for r := range in.rows {
		for j := range 8 {
			want := plainDot(in.rows[r], func(i int) float32 { return packed[8*i+j] })
			checkKernel(t, "Kernel4x8", in.n, 8*r+j, got[8*r+j], want)
		}
	}
}

func randomFloats(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

func toBF16(values []float32) []uint16 {
	out := make([]uint16, len(values))
	for i, v := range values {
		out[i] = float32ToBF16(v)
	}
	return out
}

func plainDot(x []float32, w func(i int) float32) float64 {
	var total float64
	for i, v := range x {
		total += float64(v) * float64(w(i))
	}
	return total
}

func checkKernel(t *testing.T, name string, n, at int, got float32, want float64) {
	t.Helper()
	if !approx.Equal(float64(got), want, 1e-4, 1e-4) {
		t.Errorf("%s n=%d at %d: %f, want %f", name, n, at, got, want)
	}
}

type bf16Case struct {
	bits uint32
	want uint16
}

// TestFloat32ToBF16 rounds to the nearest bf16, ties to even, as PyTorch does, and keeps NaN a NaN.
func TestFloat32ToBF16(t *testing.T) {
	for _, c := range []bf16Case{
		{0x3f808000, 0x3f80}, // a tie rounds to the even value below
		{0x3f818000, 0x3f82}, // a tie rounds to the even value above
		{0x3f807fff, 0x3f80},
		{0x3fffffff, 0x4000}, // the carry reaches the exponent
		{0x7f800000, 0x7f80},
	} {
		if got := float32ToBF16(math.Float32frombits(c.bits)); got != c.want {
			t.Errorf("float32ToBF16(%#x) = %#x, want %#x", c.bits, got, c.want)
		}
	}

	if got := bf16ToFloat32(float32ToBF16(float32(math.NaN()))); got == got {
		t.Errorf("NaN became %g", got)
	}
}

// TestLinear checks a matrix against plain float64 sums, including outputs that are not a multiple of four, as the
// decision head's single logit is, and an odd number of rows, whose last has no row to pair with.
func TestLinear(t *testing.T) {
	var shapes [][3]int
	for _, rows := range []int{1, 3, 20} {
		for _, out := range []int{1, 5, 8, 13} {
			shapes = append(shapes, [3]int{rows, out, 24})
		}
	}
	// Large enough that the pool splits it, with rows past a block and a last item of fewer than four rows of W.
	shapes = append(shapes, [3]int{70, 1027, 256})
	for width := range widths() {
		t.Run(width, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(3, 4))
			for _, shape := range shapes {
				checkLinear(t, rng, shape[0], shape[1], shape[2])
			}
		})
	}
}

// checkLinear multiplies rows random inputs of in values by a random bf16 matrix of out rows and checks each output
// against the product in float64.
func checkLinear(t *testing.T, rng *rand.Rand, rows, out, in int) {
	t.Helper()
	bf16, x := toBF16(randomFloats(rng, out*in)), randomFloats(rng, rows*in)
	got := make([]float32, rows*out)
	Linear(got, x, Matrix{cols: in, bf16: bf16})
	for i := range got {
		want := plainDot(x[i/out*in:][:in], func(j int) float32 { return bf16ToFloat32(bf16[i%out*in+j]) })

		if !approx.Equal(float64(got[i]), want, 1e-4, 1e-4) {
			t.Errorf("rows %d out %d in %d at %d: %f, want %f", rows, out, in, i, got[i], want)
		}
	}
}

// BenchmarkLinear runs two rows through each of the model's matrix shapes, out by in: paired in one call, where they
// share each load of W, and alone in a call each.
func BenchmarkLinear(b *testing.B) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, shape := range [][2]int{{384, 384}, {1152, 384}, {384, 1152}, {512, 512}, {1536, 512}, {512, 1536}} {
		out, in := shape[0], shape[1]
		w := Matrix{cols: in, bf16: toBF16(randomFloats(rng, out*in))}
		x, y := randomFloats(rng, 2*in), make([]float32, 2*out)

		b.Run(fmt.Sprintf("%dx%d/paired", out, in), func(b *testing.B) {
			for b.Loop() {
				Linear(y, x, w)
			}
		})
		b.Run(fmt.Sprintf("%dx%d/alone", out, in), func(b *testing.B) {
			for b.Loop() {
				Linear(y[:out], x[:in], w)
				Linear(y[out:], x[in:], w)
			}
		})
	}
}
