//go:build goexperiment.simd && arm64

package tensor

// wantWidths is the vector widths every arm64 CPU runs, in float32 lanes: NEON's four.
func wantWidths() []int {
	return []int{0, 4}
}
