package tensor

import (
	"maps"
	"slices"
)

// kernelSet is one vector width's kernels. Each computes what its scalar twin does up to rounding: which multiply-adds
// are fused, and the order a sum adds its terms in, may differ between them.
type kernelSet struct {
	dot                func(a, b []float32) float32
	axpy               func(y []float32, a float32, x []float32)
	kernel4x8          func(p, w0, w1, w2, w3 []float32) [32]float32
	snakeRow           func(out, x []float32, alpha float32)
	expRow             func(out, x []float32, scale, shift float32)
	dotBF16            func(x []float32, w []uint16) float32
	dot4BF16           func(x []float32, w0, w1, w2, w3 []uint16) (float32, float32, float32, float32)
	dot4x2BF16         func(x, z []float32, w0, w1, w2, w3 []uint16) ([4]float32, [4]float32)
	accumulateRowsBF16 func(out, weights []float32, rows []uint16, stride int)
	max                func(x []float32) float32
	sum                func(x []float32) float32
}

var kernelsScalar = kernelSet{
	dot:                dotScalar,
	axpy:               axpyScalar,
	kernel4x8:          kernel4x8Scalar,
	snakeRow:           snakeRowScalar,
	expRow:             expRowScalar,
	dotBF16:            dotBF16Scalar,
	dot4BF16:           dot4BF16Scalar,
	dot4x2BF16:         dot4x2BF16Scalar,
	accumulateRowsBF16: accumulateRowsBF16Scalar,
	max:                maxScalar,
	sum:                sumScalar,
}

// kernelSets holds the kernels of each vector width this CPU runs, by float32 lanes, the scalar ones at zero.
var kernelSets = map[int]kernelSet{0: kernelsScalar}

// active is the widest set, chosen when the program starts.
var active kernelSet

func init() {
	maps.Copy(kernelSets, cpuKernelSets())
	active = kernelSets[slices.Max(slices.Collect(maps.Keys(kernelSets)))]
}

// Dot is a's dot product with b.
func Dot(a, b []float32) float32 {
	return active.dot(a, b)
}

// Axpy adds a times x into y.
func Axpy(y []float32, a float32, x []float32) {
	active.axpy(y, a, x)
}

// Kernel4x8 is four rows of weights against eight packed input columns: out[8r+j] is row r's dot product with column
// j, where p holds the eight columns' values for input i at p[8i:8i+8].
func Kernel4x8(p, w0, w1, w2, w3 []float32) [32]float32 {
	return active.kernel4x8(p, w0, w1, w2, w3)
}

// SnakeRow sets out[t] to x[t] + sin(alpha x[t])^2 / alpha.
func SnakeRow(out, x []float32, alpha float32) {
	active.snakeRow(out, x, alpha)
}

// ExpRow sets out[i] to exp(x[i]*scale + shift), and to zero where that is below the smallest normal float32's log.
// An argument just below the largest float32's log or above is clamped there, so no output is infinite.
func ExpRow(out, x []float32, scale, shift float32) {
	active.expRow(out, x, scale, shift)
}

// DotBF16 is x's dot product with a bf16 row.
func DotBF16(x []float32, w []uint16) float32 {
	return active.dotBF16(x, w)
}

// Dot4BF16 is x's dot products with four bf16 rows.
func Dot4BF16(x []float32, w0, w1, w2, w3 []uint16) (float32, float32, float32, float32) {
	return active.dot4BF16(x, w0, w1, w2, w3)
}

// dot4x2BF16 is Dot4BF16 of x and of z, each equal to Dot4BF16 run on that input alone.
func dot4x2BF16(x, z []float32, w0, w1, w2, w3 []uint16) ([4]float32, [4]float32) {
	return active.dot4x2BF16(x, z, w0, w1, w2, w3)
}

// AccumulateRowsBF16 adds weights[p] times bf16 row p into out, row p starting at rows[p*stride].
func AccumulateRowsBF16(out, weights []float32, rows []uint16, stride int) {
	active.accumulateRowsBF16(out, weights, rows, stride)
}

// Max is the largest value of x, which holds no NaN, and minus infinity when x is empty.
func Max(x []float32) float32 {
	return active.max(x)
}

// Sum is the sum of x.
func Sum(x []float32) float32 {
	return active.sum(x)
}
