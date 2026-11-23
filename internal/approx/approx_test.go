package approx

import (
	"math"
	"testing"
)

// TestNaNIsNeverClose fails every comparison with NaN and slices of two lengths, and adds the two tolerances.
func TestNaNIsNeverClose(t *testing.T) {
	nan := float32(math.NaN())
	if Equal(math.NaN(), 1, 1, 1) || Equal(1, math.NaN(), 1, 1) {
		t.Error("NaN was equal to a number")
	}

	if d := MaxDifference([]float32{1, nan}, []float32{1, 2}); !math.IsInf(d, 1) {
		t.Errorf("a NaN gave a difference of %v", d)
	}

	if d := MaxDifference([]float32{1}, []float32{1, 2}); !math.IsInf(d, 1) {
		t.Errorf("slices of two lengths gave a difference of %v", d)
	}

	if !Equal(1.05, 1, 0, 0.1) || Equal(1.2, 1, 0.1, 0.05) {
		t.Error("the tolerance is not absolute plus relative")
	}
}
