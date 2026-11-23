package tensor

// Values are convolution weights in the type the release stores them in: bf16 where rounding was measured harmless,
// f32 where it was not. A convolution reuses each weight across many positions and spends its time on arithmetic rather
// than on reading its weights, so its rows widen to f32 a few at a time, where a Matrix widens inside its products.
type Values struct {
	// Exactly one of the two holds the values.
	f32  []float32
	bf16 []uint16
}

// At is value i as f32. It keeps a bf16 -0 as -0, where Float32 gives +0; both add nothing to a sum.
func (v Values) At(i int) float32 {
	if v.bf16 != nil {
		return bf16ToFloat32(v.bf16[i])
	}
	return v.f32[i]
}

// Slice is values lo to hi, sharing v's memory.
func (v Values) Slice(lo, hi int) Values {
	if v.bf16 != nil {
		return Values{bf16: v.bf16[lo:hi]}
	}
	return Values{f32: v.f32[lo:hi]}
}

// Float32 is v as f32: v's own memory when it holds f32, or its values widened into buf, which is at least as long.
func (v Values) Float32(buf []float32) []float32 {
	if v.bf16 == nil {
		return v.f32
	}

	buf = buf[:len(v.bf16)]
	widenBF16(buf, v.bf16)
	return buf
}

// Gathered is, in v's type, the value v holds at source(i) for each i in turn.
func (v Values) Gathered(source func(i int) int) Values {
	if v.bf16 != nil {
		return Values{bf16: gather(v.bf16, source)}
	}
	return Values{f32: gather(v.f32, source)}
}

func gather[T any](values []T, source func(i int) int) []T {
	out := make([]T, len(values))
	for i := range out {
		out[i] = values[source(i)]
	}
	return out
}
