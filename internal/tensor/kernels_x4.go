//go:build goexperiment.simd && (amd64 || arm64)

package tensor

import (
	"math"
	"simd/archsimd"
)

// kernelsX4 runs vectors of four float32 lanes: NEON on arm64, AVX on amd64.
var kernelsX4 = kernelSet{
	dot:                dotX4,
	axpy:               axpyX4,
	kernel4x8:          kernel4x8X4,
	snakeRow:           snakeRowX4,
	expRow:             expRowX4,
	dotBF16:            dotBF16X4,
	dot4BF16:           dot4BF16X4,
	dot4x2BF16:         dot4x2BF16X4,
	accumulateRowsBF16: accumulateRowsBF16X4,
	max:                maxX4,
	sum:                sumX4,
}

func load4(s []float32, i int) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array(arrayAt[[4]float32](s, i))
}

func store4(v archsimd.Float32x4, s []float32, i int) {
	v.StoreArray(arrayAt[[4]float32](s, i))
}

// widen4 is eight bf16 values as two f32 vectors, since the narrowest vector of uint16 holds eight: a bf16 value is the
// high half of an f32's bits.
func widen4(s []uint16, i int) (archsimd.Float32x4, archsimd.Float32x4) {
	var zero archsimd.Uint16x8
	x := archsimd.LoadUint16x8Array(arrayAt[[8]uint16](s, i))
	lo := zero.InterleaveLo(x).ReshapeToUint32s().BitsToFloat32()
	return lo, zero.InterleaveHi(x).ReshapeToUint32s().BitsToFloat32()
}

func sum4(v archsimd.Float32x4) float32 {
	var lanes [4]float32
	v.StoreArray(&lanes)
	return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

// dotX4 keeps two sums, so successive multiply-adds into one sum do not wait on each other.
func dotX4(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1 archsimd.Float32x4
	i := 0
	for ; i+8 <= len(a); i += 8 {
		s0 = load4(a, i).MulAdd(load4(b, i), s0)
		s1 = load4(a, i+4).MulAdd(load4(b, i+4), s1)
	}
	return sum4(s0.Add(s1)) + dotTail(a, b, i)
}

// axpyX4 runs the leftover steps through the vector too, padded, so a step's value does not depend on where its span
// ends. The padding is a copy on the stack: on amd64, archsimd's partial loads read a whole vector before masking it,
// past the slice's end, which faults where the slice ends at the last mapped page.
func axpyX4(y []float32, a float32, x []float32) {
	x = x[:len(y)]
	whole := len(y) &^ 3
	scale := archsimd.BroadcastFloat32x4(a)
	for i := 0; i < whole; i += 4 {
		store4(load4(x, i).MulAdd(scale, load4(y, i)), y, i)
	}

	if whole < len(y) {
		var xs, ys [4]float32
		copy(xs[:], x[whole:])
		copy(ys[:], y[whole:])
		archsimd.LoadFloat32x4Array(&xs).MulAdd(scale, archsimd.LoadFloat32x4Array(&ys)).StoreArray(&ys)
		copy(y[whole:], ys[:])
	}
}

// kernel4x8X4 holds a row's eight steps in two vectors, loading each step group once and each weight once for eight
// multiply-adds.
func kernel4x8X4(p, w0, w1, w2, w3 []float32) [32]float32 {
	n := len(w0)
	p, w1, w2, w3 = p[:8*n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x4
	for i := range n {
		u, v := load4(p, 8*i), load4(p, 8*i+4)
		s0, s1 := archsimd.BroadcastFloat32x4(w0[i]), archsimd.BroadcastFloat32x4(w1[i])
		s2, s3 := archsimd.BroadcastFloat32x4(w2[i]), archsimd.BroadcastFloat32x4(w3[i])
		a0, b0 = s0.MulAdd(u, a0), s0.MulAdd(v, b0)
		a1, b1 = s1.MulAdd(u, a1), s1.MulAdd(v, b1)
		a2, b2 = s2.MulAdd(u, a2), s2.MulAdd(v, b2)
		a3, b3 = s3.MulAdd(u, a3), s3.MulAdd(v, b3)
	}

	var out [32]float32
	for r, v := range [8]archsimd.Float32x4{a0, b0, a1, b1, a2, b2, a3, b3} {
		store4(v, out[:], 4*r)
	}
	return out
}

// snakeRowX4 sets what snakeRowScalar sets for a span of any length, its leftover steps padded as axpyX4's are.
func snakeRowX4(out, x []float32, alpha float32) {
	out = out[:len(x)]
	whole := len(x) &^ 3
	snakeGroupsX4(out[:whole], x[:whole], alpha)
	if whole < len(x) {
		var tail [4]float32
		copy(tail[:], x[whole:])
		snakeGroupsX4(tail[:], tail[:], alpha)
		copy(out[whole:], tail[:])
	}
}

func snakeGroupsX4(out, x []float32, alpha float32) {
	out = out[:len(x)]
	scale, inverse := archsimd.BroadcastFloat32x4(alpha), archsimd.BroadcastFloat32x4(1/(alpha+1e-9))
	inversePi := archsimd.BroadcastFloat32x4(1 / math.Pi)
	high, low := archsimd.BroadcastFloat32x4(piHigh), archsimd.BroadcastFloat32x4(piLow)
	// Separate variables, not an array: Go keeps an array of vectors in memory and reloads it on every pass.
	c11, c9, c7 := archsimd.BroadcastFloat32x4(sineTerms[0]), archsimd.BroadcastFloat32x4(sineTerms[1]),
		archsimd.BroadcastFloat32x4(sineTerms[2])
	c5, c3, c1 := archsimd.BroadcastFloat32x4(sineTerms[3]), archsimd.BroadcastFloat32x4(sineTerms[4]),
		archsimd.BroadcastFloat32x4(sineTerms[5])

	for t := 0; t < len(x); t += 4 {
		v := load4(x, t)
		z := v.Mul(scale)
		k := z.Mul(inversePi).Round()
		r := z.Sub(k.Mul(high)).Sub(k.Mul(low))
		r2 := r.Mul(r)
		p := c11.MulAdd(r2, c9).MulAdd(r2, c7).MulAdd(r2, c5).MulAdd(r2, c3).MulAdd(r2, c1)
		sin := r.Mul(p)
		store4(sin.Mul(sin).MulAdd(inverse, v), out, t)
	}
}

// expRowX4 pads its leftover steps as snakeRowX4 does.
func expRowX4(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	whole := len(x) &^ 3
	expGroupsX4(out[:whole], x[:whole], scale, shift)
	if whole < len(x) {
		var tail [4]float32
		copy(tail[:], x[whole:])
		expGroupsX4(tail[:], tail[:], scale, shift)
		copy(out[whole:], tail[:])
	}
}

func expGroupsX4(out, x []float32, scale, shift float32) {
	out = out[:len(x)]
	broadcast := archsimd.BroadcastFloat32x4
	scaleV, shiftV := broadcast(scale), broadcast(shift)
	low, high, one := broadcast(expLow), broadcast(expHigh), broadcast(1)
	log2e, high2, low2, bias := broadcast(log2E), broadcast(ln2High), broadcast(ln2Low), broadcast(expBias)
	c5, c4, c3 := broadcast(expTerms[0]), broadcast(expTerms[1]), broadcast(expTerms[2])
	c2, c1, c0 := broadcast(expTerms[3]), broadcast(expTerms[4]), broadcast(expTerms[5])

	for t := 0; t < len(x); t += 4 {
		z := load4(x, t).MulAdd(scaleV, shiftV)
		normal := z.GreaterEqual(low)
		z = z.Min(high)
		n := z.Mul(log2e).Round()
		r := z.Sub(n.Mul(high2)).Sub(n.Mul(low2))
		p := c5.MulAdd(r, c4).MulAdd(r, c3).MulAdd(r, c2).MulAdd(r, c1).MulAdd(r, c0)
		power := n.Add(bias).ConvertToInt32().ShiftAllLeft(expShift).ToBits().BitsToFloat32()
		store4(p.Mul(r).Mul(r).Add(r).Add(one).Mul(power).Masked(normal), out, t)
	}
}

// dotBF16X4 keeps two sums, as dotX4 does, which is how dot4BF16X4 adds a row.
func dotBF16X4(x []float32, w []uint16) float32 {
	w = w[:len(x)]
	var s0, s1 archsimd.Float32x4
	i := 0
	for ; i+8 <= len(x); i += 8 {
		lo, hi := widen4(w, i)
		s0, s1 = lo.MulAdd(load4(x, i), s0), hi.MulAdd(load4(x, i+4), s1)
	}
	return sum4(s0.Add(s1)) + dotBF16Tail(x, w, i)
}

// dot4BF16X4 is four dot products sharing each load of x, the bf16 rows widened to f32 as they are loaded. Each
// widened vector of a row has its own sum.
func dot4BF16X4(x []float32, w0, w1, w2, w3 []uint16) (float32, float32, float32, float32) {
	n := len(x)
	w0, w1, w2, w3 = w0[:n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x4
	i := 0
	for ; i+8 <= n; i += 8 {
		u, v := load4(x, i), load4(x, i+4)
		lo, hi := widen4(w0, i)
		a0, b0 = lo.MulAdd(u, a0), hi.MulAdd(v, b0)
		lo, hi = widen4(w1, i)
		a1, b1 = lo.MulAdd(u, a1), hi.MulAdd(v, b1)
		lo, hi = widen4(w2, i)
		a2, b2 = lo.MulAdd(u, a2), hi.MulAdd(v, b2)
		lo, hi = widen4(w3, i)
		a3, b3 = lo.MulAdd(u, a3), hi.MulAdd(v, b3)
	}

	t0, t1, t2, t3 := dot4BF16Tail(x, w0, w1, w2, w3, i)
	return sum4(a0.Add(b0)) + t0, sum4(a1.Add(b1)) + t1, sum4(a2.Add(b2)) + t2, sum4(a3.Add(b3)) + t3
}

// dot4x2BF16X4 is dot4BF16X4 of x and of z, each row of W widened once for both. Each input's sums add in the order
// dot4BF16X4 adds them, so pairing rows of a batch changes no result.
func dot4x2BF16X4(x, z []float32, w0, w1, w2, w3 []uint16) ([4]float32, [4]float32) {
	n := len(x)
	z, w0, w1, w2, w3 = z[:n], w0[:n], w1[:n], w2[:n], w3[:n]
	var a0, a1, a2, a3, b0, b1, b2, b3 archsimd.Float32x4
	var c0, c1, c2, c3, d0, d1, d2, d3 archsimd.Float32x4
	i := 0
	for ; i+8 <= n; i += 8 {
		u, v := load4(x, i), load4(x, i+4)
		p, q := load4(z, i), load4(z, i+4)
		lo, hi := widen4(w0, i)
		a0, b0, c0, d0 = lo.MulAdd(u, a0), hi.MulAdd(v, b0), lo.MulAdd(p, c0), hi.MulAdd(q, d0)
		lo, hi = widen4(w1, i)
		a1, b1, c1, d1 = lo.MulAdd(u, a1), hi.MulAdd(v, b1), lo.MulAdd(p, c1), hi.MulAdd(q, d1)
		lo, hi = widen4(w2, i)
		a2, b2, c2, d2 = lo.MulAdd(u, a2), hi.MulAdd(v, b2), lo.MulAdd(p, c2), hi.MulAdd(q, d2)
		lo, hi = widen4(w3, i)
		a3, b3, c3, d3 = lo.MulAdd(u, a3), hi.MulAdd(v, b3), lo.MulAdd(p, c3), hi.MulAdd(q, d3)
	}

	s0, s1, s2, s3 := dot4BF16Tail(x, w0, w1, w2, w3, i)
	t0, t1, t2, t3 := dot4BF16Tail(z, w0, w1, w2, w3, i)
	return [4]float32{sum4(a0.Add(b0)) + s0, sum4(a1.Add(b1)) + s1, sum4(a2.Add(b2)) + s2, sum4(a3.Add(b3)) + s3},
		[4]float32{sum4(c0.Add(d0)) + t0, sum4(c1.Add(d1)) + t1, sum4(c2.Add(d2)) + t2, sum4(c3.Add(d3)) + t3}
}

// accumulateRowsBF16X4 keeps a block of outputs in registers across every row, two widened loads of each row a block.
func accumulateRowsBF16X4(out, weights []float32, rows []uint16, stride int) {
	// Without a row, the length the rows are cut to below would be negative.
	if len(weights) == 0 {
		return
	}

	rows = rows[:(len(weights)-1)*stride+len(out)]
	j := 0
	for ; j+16 <= len(out); j += 16 {
		a0, a1, a2, a3 := load4(out, j), load4(out, j+4), load4(out, j+8), load4(out, j+12)
		for p, weight := range weights {
			row := p*stride + j
			scale := archsimd.BroadcastFloat32x4(weight)
			lo, hi := widen4(rows, row)
			a0, a1 = lo.MulAdd(scale, a0), hi.MulAdd(scale, a1)
			lo, hi = widen4(rows, row+8)
			a2, a3 = lo.MulAdd(scale, a2), hi.MulAdd(scale, a3)
		}
		store4(a0, out, j)
		store4(a1, out, j+4)
		store4(a2, out, j+8)
		store4(a3, out, j+12)
	}

	if j < len(out) {
		accumulateRowsBF16Scalar(out[j:], weights, rows[j:], stride)
	}
}

// maxX4 keeps a running maximum a lane; the order maxima are taken in does not change the result.
func maxX4(x []float32) float32 {
	if len(x) < 4 {
		return maxScalar(x)
	}

	m := load4(x, 0)
	i := 4
	for ; i+4 <= len(x); i += 4 {
		m = m.Max(load4(x, i))
	}

	var lanes [4]float32
	m.StoreArray(&lanes)
	return max(maxScalar(lanes[:]), maxTail(x, i))
}

// sumX4 keeps two sums, as dotX4 does.
func sumX4(x []float32) float32 {
	var s0, s1 archsimd.Float32x4
	i := 0
	for ; i+8 <= len(x); i += 8 {
		s0, s1 = s0.Add(load4(x, i)), s1.Add(load4(x, i+4))
	}
	return sum4(s0.Add(s1)) + sumTail(x, i)
}
