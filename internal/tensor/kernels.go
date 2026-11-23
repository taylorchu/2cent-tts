// Package tensor holds what every model of the engine computes with: weight matrices in bf16, the kernels that
// read them (SIMD under GOEXPERIMENT=simd, plain Go otherwise), work split over the cores, and safetensors
// loading. It depends on the standard library alone.
package tensor

import (
	"math"
)

// Sized is buf holding n values, reusing its memory when it is large enough; the values are whatever it held before.
func Sized[T any](buf *[]T, n int) []T {
	if cap(*buf) < n {
		*buf = make([]T, n)
	}
	*buf = (*buf)[:n]
	return *buf
}

// SizedPerWorker sizes a buffer of n values for each worker a parallel call may run on, making the list when it is
// nil. Sizing all of them on the caller keeps a worker that takes its first part late from allocating then.
func SizedPerWorker[T any](bufs *[][]T, n int) {
	if *bufs == nil {
		*bufs = make([][]T, workers)
	}

	SizedEach(*bufs, n)
}

// SizedEach sizes each of bufs as Sized does.
func SizedEach[T any](bufs [][]T, n int) {
	for i := range bufs {
		Sized(&bufs[i], n)
	}
}

// Sigmoid is the logistic function.
func Sigmoid(x float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(x))))
}

// ExpFromMax sets out[i] to exp((x[i] - max(x)) * scale) and returns max(x): a softmax's exponentials, the largest
// value taken off first so none overflows. out may be x.
func ExpFromMax(out, x []float32, scale float32) float32 {
	top := Max(x)
	ExpRow(out, x, scale, -top*scale)
	return top
}

// SiLURow sets out to each value of x times its logistic function; out must not be x.
func SiLURow(out, x []float32) {
	ExpRow(out, x, -1, 0)
	for i, v := range x {
		out[i] = v / (1 + out[i])
	}
}

// SigmoidRow sets out to the logistic function of each value of x; out may be x.
func SigmoidRow(out, x []float32) {
	ExpRow(out, x, -1, 0)
	for i, v := range out {
		out[i] = 1 / (1 + v)
	}
}

// Add adds y into x.
func Add(x, y []float32) {
	Axpy(x, 1, y)
}
