//go:build goexperiment.simd && amd64

package tensor

import (
	"math"
	"simd/archsimd"
)

// kernelsX8 runs vectors of eight float32 lanes: AVX2.
var kernelsX8 = kernelSet{
	dot:                dotX8,
	axpy:               axpyX8,
	kernel4x8:          kernel4x8X8,
	snakeRow:           snakeRowX8,
	expRow:             expRowX8,
	dotBF16:            dotBF16X8,
	dot4BF16:           dot4BF16X8,
	dot4x2BF16:         dot4x2BF16X8,
	accumulateRowsBF16: accumulateRowsBF16X8,
	max:                maxX8,
	sum:                sumX8,
}

func load8(s []float32, i int) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array(arrayAt[[8]float32](s, i))
}

func store8(v archsimd.Float32x8, s []float32, i int) {
	v.StoreArray(arrayAt[[8]float32](s, i))
}

// widen8 is eight bf16 values as an f32 vector: a bf16 value is the high half of an f32's bits.
func widen8(s []uint16, i int) archsimd.Float32x8 {
	return archsimd.LoadUint16x8Array(arrayAt[[8]uint16](s, i)).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
}

func sum8(v archsimd.Float32x8) float32 {
	return sum4(v.GetLo().Add(v.GetHi()))
}

// dotX8 keeps two sums, as dotX4 does.
func dotX8(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1 archsimd.Float32x8
	i := 0
	for ; i+16 <= len(a); i += 16 {
		s0 = load8(a, i).MulAdd(load8(b, i), s0)
		s1 = load8(a, i+8).MulAdd(load8(b, i+8), s1)
	}
	return sum8(s0.Add(s1)) + dotTail(a, b, i)
}

// axpyX8 pads its leftover steps as axpyX4 does.
func axpyX8(y []float32, a float32, x []float32) {
	x = x[:len(y)]
	whole := len(y) &^ 7
	scale := archsimd.BroadcastFloat32x8(a)
	for i := 0; i < whole; i += 8 {
		store8(load8(x, i).MulAdd(scale, load8(y, i)), y, i)
	}

	if whole < len(y) {
		var xs, ys [8]float32
		copy(xs[:], x[whole:])
		copy(ys[:], y[whole:])
		archsimd.LoadFloat32x8Array(&xs).MulAdd(scale, archsimd.LoadFloat32x8Array(&ys)).StoreArray(&ys)
		copy(y[whole:], ys[:])
	}
}

// kernel4x8X8 holds a row's eight steps in one vector.
func kernel4x8X8(p, w0, w1, w2, w3 []float32) [32]float32 {
	n := len(w0)
	p, w1, w2, w3 = p[:8*n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3 archsimd.Float32x8
	for i := range n {
		u := load8(p, 8*i)
		a0 = archsimd.BroadcastFloat32x8(w0[i]).MulAdd(u, a0)
		a1 = archsimd.BroadcastFloat32x8(w1[i]).MulAdd(u, a1)
		a2 = archsimd.BroadcastFloat32x8(w2[i]).MulAdd(u, a2)
		a3 = archsimd.BroadcastFloat32x8(w3[i]).MulAdd(u, a3)
	}

	var out [32]float32
	for r, v := range [4]archsimd.Float32x8{a0, a1, a2, a3} {
		store8(v, out[:], 8*r)
	}
	return out
}

// snakeRowX8 pads its leftover steps as snakeRowX4 does.
func snakeRowX8(out, x []float32, alpha float32) {
	out = out[:len(x)]
	whole := len(x) &^ 7
	snakeGroupsX8(out[:whole], x[:whole], alpha)
	if whole < len(x) {
		var tail [8]float32
		copy(tail[:], x[whole:])
		snakeGroupsX8(tail[:], tail[:], alpha)
		copy(out[whole:], tail[:])
	}
}

func snakeGroupsX8(out, x []float32, alpha float32) {
	out = out[:len(x)]
	scale, inverse := archsimd.BroadcastFloat32x8(alpha), archsimd.BroadcastFloat32x8(1/(alpha+1e-9))
	inversePi := archsimd.BroadcastFloat32x8(1 / math.Pi)
	high, low := archsimd.BroadcastFloat32x8(piHigh), archsimd.BroadcastFloat32x8(piLow)
	c11, c9, c7 := archsimd.BroadcastFloat32x8(sineTerms[0]), archsimd.BroadcastFloat32x8(sineTerms[1]),
		archsimd.BroadcastFloat32x8(sineTerms[2])
	c5, c3, c1 := archsimd.BroadcastFloat32x8(sineTerms[3]), archsimd.BroadcastFloat32x8(sineTerms[4]),
		archsimd.BroadcastFloat32x8(sineTerms[5])

	for t := 0; t < len(x); t += 8 {
		v := load8(x, t)
		z := v.Mul(scale)
		k := z.Mul(inversePi).Round()
		r := z.Sub(k.Mul(high)).Sub(k.Mul(low))
		r2 := r.Mul(r)
		p := c11.MulAdd(r2, c9).MulAdd(r2, c7).MulAdd(r2, c5).MulAdd(r2, c3).MulAdd(r2, c1)
		sin := r.Mul(p)
		store8(sin.Mul(sin).MulAdd(inverse, v), out, t)
	}
}

// expRowX8 pads its leftover steps as snakeRowX4 does.
func expRowX8(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	whole := len(x) &^ 7
	expGroupsX8(out[:whole], x[:whole], scale, shift)
	if whole < len(x) {
		var tail [8]float32
		copy(tail[:], x[whole:])
		expGroupsX8(tail[:], tail[:], scale, shift)
		copy(out[whole:], tail[:])
	}
}

func expGroupsX8(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	broadcast := archsimd.BroadcastFloat32x8
	scaleV, shiftV := broadcast(scale), broadcast(shift)
	low, high, one := broadcast(expLow), broadcast(expHigh), broadcast(1)
	log2e, high2, low2, bias := broadcast(log2E), broadcast(ln2High), broadcast(ln2Low), broadcast(expBias)
	c5, c4, c3 := broadcast(expTerms[0]), broadcast(expTerms[1]), broadcast(expTerms[2])
	c2, c1, c0 := broadcast(expTerms[3]), broadcast(expTerms[4]), broadcast(expTerms[5])

	for t := 0; t < len(x); t += 8 {
		z := load8(x, t).MulAdd(scaleV, shiftV)
		normal := z.GreaterEqual(low)
		z = z.Min(high)
		n := z.Mul(log2e).Round()
		r := z.Sub(n.Mul(high2)).Sub(n.Mul(low2))
		p := c5.MulAdd(r, c4).MulAdd(r, c3).MulAdd(r, c2).MulAdd(r, c1).MulAdd(r, c0)
		power := n.Add(bias).ConvertToInt32().ShiftAllLeft(expShift).ToBits().BitsToFloat32()
		store8(p.Mul(r).Mul(r).Add(r).Add(one).Mul(power).Masked(normal), out, t)
	}
}

// dotBF16X8 adds as dot4BF16X8 adds a row, so a dot product is the same wherever the engine computes it.
func dotBF16X8(x []float32, w []uint16) float32 {
	w = w[:len(x)]
	var s archsimd.Float32x8
	i := 0
	for ; i+8 <= len(x); i += 8 {
		s = widen8(w, i).MulAdd(load8(x, i), s)
	}
	return sum8(s) + dotBF16Tail(x, w, i)
}

// dot4BF16X8 gives each widened vector of a row its own sum, as dot4BF16X4 does.
func dot4BF16X8(x []float32, w0, w1, w2, w3 []uint16) (float32, float32, float32, float32) {
	n := len(x)
	w0, w1, w2, w3 = w0[:n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		u := load8(x, i)
		a0, a1 = widen8(w0, i).MulAdd(u, a0), widen8(w1, i).MulAdd(u, a1)
		a2, a3 = widen8(w2, i).MulAdd(u, a2), widen8(w3, i).MulAdd(u, a3)
	}

	t0, t1, t2, t3 := dot4BF16Tail(x, w0, w1, w2, w3, i)
	return sum8(a0) + t0, sum8(a1) + t1, sum8(a2) + t2, sum8(a3) + t3
}

// dot4x2BF16X8 adds each input's sums in the order dot4BF16X8 adds them, as dot4x2BF16X4 does.
func dot4x2BF16X8(x, z []float32, w0, w1, w2, w3 []uint16) ([4]float32, [4]float32) {
	n := len(x)
	z, w0, w1, w2, w3 = z[:n], w0[:n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3, c0, c1, c2, c3 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		u, p := load8(x, i), load8(z, i)
		w := widen8(w0, i)
		a0, c0 = w.MulAdd(u, a0), w.MulAdd(p, c0)
		w = widen8(w1, i)
		a1, c1 = w.MulAdd(u, a1), w.MulAdd(p, c1)
		w = widen8(w2, i)
		a2, c2 = w.MulAdd(u, a2), w.MulAdd(p, c2)
		w = widen8(w3, i)
		a3, c3 = w.MulAdd(u, a3), w.MulAdd(p, c3)
	}

	s0, s1, s2, s3 := dot4BF16Tail(x, w0, w1, w2, w3, i)
	t0, t1, t2, t3 := dot4BF16Tail(z, w0, w1, w2, w3, i)
	return [4]float32{sum8(a0) + s0, sum8(a1) + s1, sum8(a2) + s2, sum8(a3) + s3},
		[4]float32{sum8(c0) + t0, sum8(c1) + t1, sum8(c2) + t2, sum8(c3) + t3}
}

// accumulateRowsBF16X8 keeps a block of outputs as accumulateRowsBF16X4 does.
func accumulateRowsBF16X8(out, weights []float32, rows []uint16, stride int) {
	if len(weights) == 0 {
		return
	}

	rows = rows[:(len(weights)-1)*stride+len(out)]
	j := 0
	for ; j+16 <= len(out); j += 16 {
		a0, a1 := load8(out, j), load8(out, j+8)
		for p, weight := range weights {
			row := p*stride + j
			scale := archsimd.BroadcastFloat32x8(weight)
			a0, a1 = widen8(rows, row).MulAdd(scale, a0), widen8(rows, row+8).MulAdd(scale, a1)
		}
		store8(a0, out, j)
		store8(a1, out, j+8)
	}

	if j < len(out) {
		accumulateRowsBF16Scalar(out[j:], weights, rows[j:], stride)
	}
}

// maxX8 keeps a running maximum a lane; the order maxima are taken in does not change the result.
func maxX8(x []float32) float32 {
	if len(x) < 8 {
		return maxScalar(x)
	}

	m := load8(x, 0)
	i := 8
	for ; i+8 <= len(x); i += 8 {
		m = m.Max(load8(x, i))
	}

	var lanes [8]float32
	m.StoreArray(&lanes)
	return max(maxScalar(lanes[:]), maxTail(x, i))
}

// sumX8 keeps two sums, as dotX4 does.
func sumX8(x []float32) float32 {
	var s0, s1 archsimd.Float32x8
	i := 0
	for ; i+16 <= len(x); i += 16 {
		s0, s1 = s0.Add(load8(x, i)), s1.Add(load8(x, i+8))
	}
	return sum8(s0.Add(s1)) + sumTail(x, i)
}
