//go:build goexperiment.simd && arm64

package tensor

// cpuKernelSets is the four-lane kernels: an arm64 vector holds four float32 lanes, with fused multiply-add on every
// CPU.
func cpuKernelSets() map[int]kernelSet {
	return map[int]kernelSet{4: kernelsX4}
}
