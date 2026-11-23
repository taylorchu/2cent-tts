//go:build goexperiment.simd && (amd64 || arm64)

package tensor

import (
	"math"
	"unsafe"
)

// arrayAt points at the array A that starts at s[i], without a bounds check. Each kernel checks its slice lengths once
// before its loop; checking every load instead cost more instructions than the multiply-adds.
func arrayAt[A, E any](s []E, i int) *A {
	var e E
	return (*A)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(s)), i*int(unsafe.Sizeof(e))))
}

// The tails are the scalar twins from i on, called only when something is left: the engine's rows are multiples of
// every vector width, and a call per kernel cost a few percent of a step.
func dotTail(a, b []float32, i int) float32 {
	if i == len(a) {
		return 0
	}
	return dotScalar(a[i:], b[i:])
}

func dotBF16Tail(x []float32, w []uint16, i int) float32 {
	if i == len(x) {
		return 0
	}
	return dotBF16Scalar(x[i:], w[i:])
}

func dot4BF16Tail(x []float32, w0, w1, w2, w3 []uint16, i int) (float32, float32, float32, float32) {
	if i == len(x) {
		return 0, 0, 0, 0
	}
	return dot4BF16Scalar(x[i:], w0[i:], w1[i:], w2[i:], w3[i:])
}

func maxTail(x []float32, i int) float32 {
	if i == len(x) {
		return float32(math.Inf(-1))
	}
	return maxScalar(x[i:])
}

func sumTail(x []float32, i int) float32 {
	if i == len(x) {
		return 0
	}
	return sumScalar(x[i:])
}
