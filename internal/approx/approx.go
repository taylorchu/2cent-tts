// Package approx compares computed values with expected ones within a tolerance. Every comparison fails on NaN, which
// an ordered comparison such as diff > tolerance lets through.
package approx

import "math"

// Equal reports whether got is within absolute plus relative times |want| of want.
func Equal(got, want, absolute, relative float64) bool {
	return math.Abs(got-want) <= absolute+relative*math.Abs(want)
}

// MaxDifference is the largest absolute difference between a and b, value by value, or infinity when their lengths
// differ or a difference is NaN.
func MaxDifference(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	worst := 0.0
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if math.IsNaN(d) {
			return math.Inf(1)
		}
		worst = max(worst, d)
	}
	return worst
}
