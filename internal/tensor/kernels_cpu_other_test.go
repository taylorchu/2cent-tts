//go:build !(goexperiment.simd && (amd64 || arm64))

package tensor

// wantWidths is the scalar kernels alone, which every build without SIMD runs.
func wantWidths() []int {
	return []int{0}
}
