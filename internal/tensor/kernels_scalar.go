package tensor

import "math"

func dotScalar(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}

	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func axpyScalar(y []float32, a float32, x []float32) {
	x = x[:len(y)]
	i := 0
	for ; i+4 <= len(y); i += 4 {
		y[i] += a * x[i]
		y[i+1] += a * x[i+1]
		y[i+2] += a * x[i+2]
		y[i+3] += a * x[i+3]
	}

	for ; i < len(y); i++ {
		y[i] += a * x[i]
	}
}

// kernel4x8Scalar is the 4 by 8 output block of rows w0..w3 against eight steps packed as p[8i:8i+8] for input i,
// as two 4 by 4 halves so the sums fit in registers.
func kernel4x8Scalar(p, w0, w1, w2, w3 []float32) [32]float32 {
	var out [32]float32
	// A tile of no columns has an empty p, which the halves below cannot slice.
	if len(w0) == 0 {
		return out
	}

	for half := range 2 {
		block := kernel4x4Strided(p[4*half:], w0, w1, w2, w3)
		for r := range 4 {
			copy(out[8*r+4*half:][:4], block[4*r:][:4])
		}
	}
	return out
}

// kernel4x4Strided is the 4 by 4 output block of rows w0..w3 against four steps at p[8i:8i+4] for input i.
func kernel4x4Strided(p, w0, w1, w2, w3 []float32) [16]float32 {
	n := len(w0)
	w1, w2, w3 = w1[:n], w2[:n], w3[:n]
	var a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33 float32
	for i := range n {
		q := p[8*i : 8*i+4 : 8*i+4]
		x0, x1, x2, x3 := q[0], q[1], q[2], q[3]
		v := w0[i]
		a00 += v * x0
		a01 += v * x1
		a02 += v * x2
		a03 += v * x3
		v = w1[i]
		a10 += v * x0
		a11 += v * x1
		a12 += v * x2
		a13 += v * x3
		v = w2[i]
		a20 += v * x0
		a21 += v * x1
		a22 += v * x2
		a23 += v * x3
		v = w3[i]
		a30 += v * x0
		a31 += v * x1
		a32 += v * x2
		a33 += v * x3
	}
	return [16]float32{a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33}
}

// piHigh has few significant bits, so k times piHigh is exact for every fold count a snake argument reaches.
const (
	piHigh = 3.140625
	piLow  = 9.67653589793e-4
)

// Taylor coefficients of sin(r)/r in r squared, accurate to about 1e-7 for |r| up to pi/2.
var sineTerms = [...]float32{-1.0 / 39916800, 1.0 / 362880, -1.0 / 5040, 1.0 / 120, -1.0 / 6, 1}

// snakeRowScalar folds the argument into [-pi/2, pi/2]: the square of the sine repeats every pi, so the sign of the
// fold drops out.
func snakeRowScalar(out, x []float32, alpha float32) {
	inverse := 1 / (alpha + 1e-9)
	for t, v := range x {
		z := alpha * v
		k := float32(math.RoundToEven(float64(z * (1 / math.Pi))))
		r := (z - k*piHigh) - k*piLow
		r2 := r * r
		p := sineTerms[0]
		for _, term := range sineTerms[1:] {
			p = p*r2 + term
		}
		sin := r * p
		out[t] = v + inverse*sin*sin
	}
}

// The exp kernels split an argument into n ln 2 + r, |r| <= ln 2 / 2, and scale a polynomial in r by 2^n. ln2High has
// few significant bits, so n times ln2High is exact. Below expLow, 2^n is no longer a normal float32; expHigh is the
// largest argument whose n rounds to 127, the largest exponent a float32 holds.
const (
	expLow   = -87.33654
	expHigh  = 88.37626
	log2E    = 1.442695
	ln2High  = 0.693359375
	ln2Low   = -2.12194440e-4
	expBias  = 127
	expShift = 23
)

var expTerms = [...]float32{1.9875691500e-4, 1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1,
	5.0000001201e-1}

func expRowScalar(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	for i, v := range x {
		z := v*scale + shift
		if !(z >= expLow) {
			out[i] = 0
			continue
		}

		z = min(z, expHigh)
		n := float32(math.RoundToEven(float64(z * log2E)))
		r := (z - n*ln2High) - n*ln2Low
		p := expTerms[0]
		for _, term := range expTerms[1:] {
			p = p*r + term
		}
		out[i] = (p*r*r + r + 1) * math.Float32frombits(uint32(n+expBias)<<expShift)
	}
}

func dotBF16Scalar(x []float32, w []uint16) float32 {
	w = w[:len(x)]
	var total float32
	for i, v := range x {
		total += v * bf16ToFloat32(w[i])
	}
	return total
}

func dot4BF16Scalar(x []float32, w0, w1, w2, w3 []uint16) (float32, float32, float32, float32) {
	w0, w1, w2, w3 = w0[:len(x)], w1[:len(x)], w2[:len(x)], w3[:len(x)]
	var a0, a1, a2, a3 float32
	for i, v := range x {
		a0 += v * bf16ToFloat32(w0[i])
		a1 += v * bf16ToFloat32(w1[i])
		a2 += v * bf16ToFloat32(w2[i])
		a3 += v * bf16ToFloat32(w3[i])
	}
	return a0, a1, a2, a3
}

func dot4x2BF16Scalar(x, z []float32, w0, w1, w2, w3 []uint16) ([4]float32, [4]float32) {
	z = z[:len(x)]
	var first, second [4]float32
	first[0], first[1], first[2], first[3] = dot4BF16Scalar(x, w0, w1, w2, w3)
	second[0], second[1], second[2], second[3] = dot4BF16Scalar(z, w0, w1, w2, w3)
	return first, second
}

func maxScalar(x []float32) float32 {
	best := float32(math.Inf(-1))
	for _, v := range x {
		best = max(best, v)
	}
	return best
}

func sumScalar(x []float32) float32 {
	var total float32
	for _, v := range x {
		total += v
	}
	return total
}

func accumulateRowsBF16Scalar(out, weights []float32, rows []uint16, stride int) {
	for p, weight := range weights {
		row := rows[p*stride:][:len(out)]
		for j := range out {
			out[j] += weight * bf16ToFloat32(row[j])
		}
	}
}
