//go:build linux || darwin

package tensor

import (
	"syscall"
	"testing"
	"unsafe"
)

// TestAxpyStopsAtTheSliceEnd runs Axpy on slices that end where readable memory ends, at every vector width and with
// every count of leftover steps: a kernel that reads past a slice's last value faults.
func TestAxpyStopsAtTheSliceEnd(t *testing.T) {
	page := syscall.Getpagesize()
	memory, err := syscall.Mmap(-1, 0, 4*page, syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("no anonymous memory map: %v", err)
	}
	defer func() { _ = syscall.Munmap(memory) }()

	// x ends before the second page, y before the fourth, and the pages after them cannot be read.
	if err := syscall.Mprotect(memory[page:2*page], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}

	if err := syscall.Mprotect(memory[3*page:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}

	floats := page / 4
	xPage := unsafe.Slice((*float32)(unsafe.Pointer(&memory[0])), floats)
	yPage := unsafe.Slice((*float32)(unsafe.Pointer(&memory[2*page])), floats)
	for name := range widths() {
		for n := 1; n <= 33; n++ {
			x, y := xPage[floats-n:], yPage[floats-n:]
			for i := range n {
				x[i], y[i] = float32(i), 1
			}

			Axpy(y, 2, x)
			for i := range n {
				if y[i] != 1+2*float32(i) {
					t.Fatalf("%s, %d steps: y[%d] = %v, want %v", name, n, i, y[i], 1+2*float32(i))
				}
			}
		}
	}
}
