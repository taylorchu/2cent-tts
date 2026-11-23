//go:build goexperiment.simd && amd64

package tensor

import "simd/archsimd"

// wantWidths is the vector widths this CPU runs, in float32 lanes: four with AVX, eight with AVX2, sixteen with
// AVX-512, each needing fused multiply-add.
func wantWidths() []int {
	want := []int{0}
	for lanes, ok := range map[int]bool{4: archsimd.X86.AVX(), 8: archsimd.X86.AVX2(), 16: archsimd.X86.AVX512()} {
		if archsimd.X86.FMA() && ok {
			want = append(want, lanes)
		}
	}
	return want
}
