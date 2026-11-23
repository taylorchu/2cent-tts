//go:build goexperiment.simd && amd64

package tensor

import (
	"math"
	"simd/archsimd"
)

// kernelsX16 runs vectors of sixteen float32 lanes: AVX-512. The packed kernel keeps eight lanes, since the convolution
// packs its steps in groups of eight. A vector of sixteen has no Round, so the snake and exp round with RoundScaled(0),
// which also rounds to the nearest integer, ties to even.
var kernelsX16 = kernelSet{
	dot:                dotX16,
	axpy:               axpyX16,
	kernel4x8:          kernel4x8X8,
	snakeRow:           snakeRowX16,
	expRow:             expRowX16,
	dotBF16:            dotBF16X16,
	dot4BF16:           dot4BF16X16,
	dot4x2BF16:         dot4x2BF16X16,
	accumulateRowsBF16: accumulateRowsBF16X16,
	max:                maxX16,
	sum:                sumX16,
}

func load16(s []float32, i int) archsimd.Float32x16 {
	return archsimd.LoadFloat32x16Array(arrayAt[[16]float32](s, i))
}

func store16(v archsimd.Float32x16, s []float32, i int) {
	v.StoreArray(arrayAt[[16]float32](s, i))
}

// widen16 is sixteen bf16 values as an f32 vector: a bf16 value is the high half of an f32's bits.
func widen16(s []uint16, i int) archsimd.Float32x16 {
	return archsimd.LoadUint16x16Array(arrayAt[[16]uint16](s, i)).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
}

func sum16(v archsimd.Float32x16) float32 {
	return sum8(v.GetLo().Add(v.GetHi()))
}

// dotX16 keeps two sums, as dotX4 does.
func dotX16(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1 archsimd.Float32x16
	i := 0
	for ; i+32 <= len(a); i += 32 {
		s0 = load16(a, i).MulAdd(load16(b, i), s0)
		s1 = load16(a, i+16).MulAdd(load16(b, i+16), s1)
	}
	return sum16(s0.Add(s1)) + dotTail(a, b, i)
}

// axpyX16 pads its leftover steps as axpyX4 does.
func axpyX16(y []float32, a float32, x []float32) {
	x = x[:len(y)]
	whole := len(y) &^ 15
	scale := archsimd.BroadcastFloat32x16(a)
	for i := 0; i < whole; i += 16 {
		store16(load16(x, i).MulAdd(scale, load16(y, i)), y, i)
	}

	if whole < len(y) {
		var xs, ys [16]float32
		copy(xs[:], x[whole:])
		copy(ys[:], y[whole:])
		archsimd.LoadFloat32x16Array(&xs).MulAdd(scale, archsimd.LoadFloat32x16Array(&ys)).StoreArray(&ys)
		copy(y[whole:], ys[:])
	}
}

// snakeRowX16 pads its leftover steps as snakeRowX4 does.
func snakeRowX16(out, x []float32, alpha float32) {
	out = out[:len(x)]
	whole := len(x) &^ 15
	snakeGroupsX16(out[:whole], x[:whole], alpha)
	if whole < len(x) {
		var tail [16]float32
		copy(tail[:], x[whole:])
		snakeGroupsX16(tail[:], tail[:], alpha)
		copy(out[whole:], tail[:])
	}
}

func snakeGroupsX16(out, x []float32, alpha float32) {
	out = out[:len(x)]
	scale, inverse := archsimd.BroadcastFloat32x16(alpha), archsimd.BroadcastFloat32x16(1/(alpha+1e-9))
	inversePi := archsimd.BroadcastFloat32x16(1 / math.Pi)
	high, low := archsimd.BroadcastFloat32x16(piHigh), archsimd.BroadcastFloat32x16(piLow)
	c11, c9, c7 := archsimd.BroadcastFloat32x16(sineTerms[0]), archsimd.BroadcastFloat32x16(sineTerms[1]),
		archsimd.BroadcastFloat32x16(sineTerms[2])
	c5, c3, c1 := archsimd.BroadcastFloat32x16(sineTerms[3]), archsimd.BroadcastFloat32x16(sineTerms[4]),
		archsimd.BroadcastFloat32x16(sineTerms[5])

	for t := 0; t < len(x); t += 16 {
		v := load16(x, t)
		z := v.Mul(scale)
		k := z.Mul(inversePi).RoundScaled(0)
		r := z.Sub(k.Mul(high)).Sub(k.Mul(low))
		r2 := r.Mul(r)
		p := c11.MulAdd(r2, c9).MulAdd(r2, c7).MulAdd(r2, c5).MulAdd(r2, c3).MulAdd(r2, c1)
		sin := r.Mul(p)
		store16(sin.Mul(sin).MulAdd(inverse, v), out, t)
	}
}

// expRowX16 pads its leftover steps as snakeRowX4 does.
func expRowX16(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	whole := len(x) &^ 15
	expGroupsX16(out[:whole], x[:whole], scale, shift)
	if whole < len(x) {
		var tail [16]float32
		copy(tail[:], x[whole:])
		expGroupsX16(tail[:], tail[:], scale, shift)
		copy(out[whole:], tail[:])
	}
}

func expGroupsX16(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	broadcast := archsimd.BroadcastFloat32x16
	scaleV, shiftV := broadcast(scale), broadcast(shift)
	low, high, one := broadcast(expLow), broadcast(expHigh), broadcast(1)
	log2e, high2, low2, bias := broadcast(log2E), broadcast(ln2High), broadcast(ln2Low), broadcast(expBias)
	c5, c4, c3 := broadcast(expTerms[0]), broadcast(expTerms[1]), broadcast(expTerms[2])
	c2, c1, c0 := broadcast(expTerms[3]), broadcast(expTerms[4]), broadcast(expTerms[5])

	for t := 0; t < len(x); t += 16 {
		z := load16(x, t).MulAdd(scaleV, shiftV)
		normal := z.GreaterEqual(low)
		z = z.Min(high)
		n := z.Mul(log2e).RoundScaled(0)
		r := z.Sub(n.Mul(high2)).Sub(n.Mul(low2))
		p := c5.MulAdd(r, c4).MulAdd(r, c3).MulAdd(r, c2).MulAdd(r, c1).MulAdd(r, c0)
		power := n.Add(bias).ConvertToInt32().ShiftAllLeft(expShift).ToBits().BitsToFloat32()
		store16(p.Mul(r).Mul(r).Add(r).Add(one).Mul(power).Masked(normal), out, t)
	}
}

// dotBF16X16 adds as dot4BF16X16 adds a row, as dotBF16X8 does.
func dotBF16X16(x []float32, w []uint16) float32 {
	w = w[:len(x)]
	var s archsimd.Float32x16
	i := 0
	for ; i+16 <= len(x); i += 16 {
		s = widen16(w, i).MulAdd(load16(x, i), s)
	}
	return sum16(s) + dotBF16Tail(x, w, i)
}

// dot4BF16X16 gives each widened vector of a row its own sum, as dot4BF16X4 does.
func dot4BF16X16(x []float32, w0, w1, w2, w3 []uint16) (float32, float32, float32, float32) {
	n := len(x)
	w0, w1, w2, w3 = w0[:n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3 archsimd.Float32x16
	i := 0
	for ; i+16 <= n; i += 16 {
		u := load16(x, i)
		a0, a1 = widen16(w0, i).MulAdd(u, a0), widen16(w1, i).MulAdd(u, a1)
		a2, a3 = widen16(w2, i).MulAdd(u, a2), widen16(w3, i).MulAdd(u, a3)
	}

	t0, t1, t2, t3 := dot4BF16Tail(x, w0, w1, w2, w3, i)
	return sum16(a0) + t0, sum16(a1) + t1, sum16(a2) + t2, sum16(a3) + t3
}

// dot4x2BF16X16 adds each input's sums in the order dot4BF16X16 adds them, as dot4x2BF16X4 does.
func dot4x2BF16X16(x, z []float32, w0, w1, w2, w3 []uint16) ([4]float32, [4]float32) {
	n := len(x)
	z, w0, w1, w2, w3 = z[:n], w0[:n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3, c0, c1, c2, c3 archsimd.Float32x16
	i := 0
	for ; i+16 <= n; i += 16 {
		u, p := load16(x, i), load16(z, i)
		w := widen16(w0, i)
		a0, c0 = w.MulAdd(u, a0), w.MulAdd(p, c0)
		w = widen16(w1, i)
		a1, c1 = w.MulAdd(u, a1), w.MulAdd(p, c1)
		w = widen16(w2, i)
		a2, c2 = w.MulAdd(u, a2), w.MulAdd(p, c2)
		w = widen16(w3, i)
		a3, c3 = w.MulAdd(u, a3), w.MulAdd(p, c3)
	}

	s0, s1, s2, s3 := dot4BF16Tail(x, w0, w1, w2, w3, i)
	t0, t1, t2, t3 := dot4BF16Tail(z, w0, w1, w2, w3, i)
	return [4]float32{sum16(a0) + s0, sum16(a1) + s1, sum16(a2) + s2, sum16(a3) + s3},
		[4]float32{sum16(c0) + t0, sum16(c1) + t1, sum16(c2) + t2, sum16(c3) + t3}
}

// accumulateRowsBF16X16 keeps a block of outputs as accumulateRowsBF16X4 does.
func accumulateRowsBF16X16(out, weights []float32, rows []uint16, stride int) {
	if len(weights) == 0 {
		return
	}

	rows = rows[:(len(weights)-1)*stride+len(out)]
	j := 0
	for ; j+32 <= len(out); j += 32 {
		a0, a1 := load16(out, j), load16(out, j+16)
		for p, weight := range weights {
			row := p*stride + j
			scale := archsimd.BroadcastFloat32x16(weight)
			a0, a1 = widen16(rows, row).MulAdd(scale, a0), widen16(rows, row+16).MulAdd(scale, a1)
		}
		store16(a0, out, j)
		store16(a1, out, j+16)
	}

	if j < len(out) {
		accumulateRowsBF16Scalar(out[j:], weights, rows[j:], stride)
	}
}

// maxX16 keeps a running maximum a lane; the order maxima are taken in does not change the result.
func maxX16(x []float32) float32 {
	if len(x) < 16 {
		return maxScalar(x)
	}

	m := load16(x, 0)
	i := 16
	for ; i+16 <= len(x); i += 16 {
		m = m.Max(load16(x, i))
	}

	var lanes [16]float32
	m.StoreArray(&lanes)
	return max(maxScalar(lanes[:]), maxTail(x, i))
}

// sumX16 keeps two sums, as dotX4 does.
func sumX16(x []float32) float32 {
	var s0, s1 archsimd.Float32x16
	i := 0
	for ; i+32 <= len(x); i += 32 {
		s0, s1 = s0.Add(load16(x, i)), s1.Add(load16(x, i+16))
	}
	return sum16(s0.Add(s1)) + sumTail(x, i)
}
