//go:build !(goexperiment.simd && (amd64 || arm64))

package tensor

// cpuKernelSets is none: without GOEXPERIMENT=simd, or on another architecture, the engine runs its scalar kernels.
func cpuKernelSets() map[int]kernelSet {
	return nil
}
