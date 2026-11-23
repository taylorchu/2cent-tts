//go:build goexperiment.simd && amd64

package tensor

import "simd/archsimd"

// cpuKernelSets is the kernels of each vector width this CPU runs: four lanes with AVX, eight with AVX2 and sixteen
// with AVX-512, each with fused multiply-add.
func cpuKernelSets() map[int]kernelSet {
	sets := map[int]kernelSet{}
	if !archsimd.X86.FMA() {
		return sets
	}

	if archsimd.X86.AVX() {
		sets[4] = kernelsX4
	}

	if archsimd.X86.AVX2() {
		sets[8] = kernelsX8
	}

	if archsimd.X86.AVX512() {
		sets[16] = kernelsX16
	}
	return sets
}
