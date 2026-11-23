package tensor

import (
	"math"
)

// Matrix holds a weight matrix's rows one after another in bf16, since generating a token reads every weight once
// and the bytes read set the speed. A Matrix is only read, so any number of goroutines may share it; the zero Matrix
// holds no rows and is not for use. A Loader makes one.
type Matrix struct {
	cols int
	bf16 []uint16
}

// RowCount is how many rows the matrix holds.
func (m Matrix) RowCount() int {
	return len(m.bf16) / m.cols
}

// Rows is rows lo to hi, sharing the matrix's values.
func (m Matrix) Rows(lo, hi int) Matrix {
	return Matrix{cols: m.cols, bf16: m.bf16[lo*m.cols : hi*m.cols]}
}

// Row writes row r into out as f32.
func (m Matrix) Row(out []float32, r int) {
	widenBF16(out[:m.cols], m.bf16[r*m.cols:])
}

// widenBF16 sets out to the first len(out) values of in as f32, with the SIMD kernel that sums rows: 1 times a bf16
// value plus zero is exactly its f32, except -0, which becomes +0.
func widenBF16(out []float32, in []uint16) {
	clear(out)
	AccumulateRowsBF16(out, unitWeight, in, len(out))
}

// unitWeight is the single row weight widenBF16 sums with.
var unitWeight = []float32{1}

func bf16ToFloat32(v uint16) float32 {
	return math.Float32frombits(uint32(v) << 16)
}

func float32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	// Rounding a NaN's payload could carry it into infinity; any NaN stays a quiet NaN instead.
	if v != v {
		return uint16(bits>>16) | 0x40
	}
	return uint16((bits + 0x7fff + (bits>>16)&1) >> 16)
}

// AppendBF16 appends values to out as bf16, rounding to nearest even.
func AppendBF16(out []uint16, values []float32) []uint16 {
	for _, v := range values {
		out = append(out, float32ToBF16(v))
	}
	return out
}

// Linear sets y[t] = W x[t] for each row of x, a row of x holding as many values as W has columns. Four rows of W at
// a time share each load of x, and two rows of x share each load of W.
func Linear(y, x []float32, w Matrix) {
	var task *linearTask
	select {
	case task = <-linearTasks:
	default:
		task = new(linearTask)
	}

	// An item is four rows of W, each multiplied with every row of x.
	RunParallel(task, linearTask{y: y, x: x, w: w}, (w.RowCount()+3)/4, 4*len(x))
	select {
	case linearTasks <- task:
	default:
	}
}

// LinearInto is Linear into buf's memory, sized to W's outputs for each row of x.
func LinearInto(buf *[]float32, x []float32, w Matrix) []float32 {
	y := Sized(buf, len(x)/w.cols*w.RowCount())
	Linear(y, x, w)
	return y
}

// linearTasks keeps Linear's calls from allocating: a call's arguments travel to the pool in a task taken from here,
// and a call that finds none free makes one. Linear is a free function called from many owners, so its tasks wait
// here rather than in each owner's memory, in a channel, which unlike a sync.Pool keeps them across collections.
var linearTasks = make(chan *linearTask, MaxParallelCalls)

type linearTask struct {
	y, x []float32
	w    Matrix
}

// Run computes rows 4*lo to 4*hi of W.
func (task *linearTask) Run(lo, hi, _ int) {
	y, x, w := task.y, task.x, task.w
	in, out := w.cols, w.RowCount()
	rows := len(x) / in
	// Rows of x go a block at a time, so the block stays in cache while every row of W in the part passes over it.
	const block = 32
	for t0 := 0; t0 < rows; t0 += block {
		t1 := min(t0+block, rows)
		for o := 4 * lo; o < min(4*hi, out); o += 4 {
			ws := w.bf16[o*in:]
			if o+4 > out {
				for t := t0; t < t1; t++ {
					for r := o; r < out; r++ {
						y[t*out+r] = DotBF16(x[t*in:][:in], ws[(r-o)*in:][:in])
					}
				}
				continue
			}

			w0, w1, w2, w3 := ws[:in], ws[in:][:in], ws[2*in:][:in], ws[3*in:][:in]
			t := t0
			for ; t+2 <= t1; t += 2 {
				first, second := dot4x2BF16(x[t*in:][:in], x[(t+1)*in:][:in], w0, w1, w2, w3)
				copy(y[t*out+o:][:4], first[:])
				copy(y[(t+1)*out+o:][:4], second[:])
			}

			if t < t1 {
				ys := y[t*out+o:][:4]
				ys[0], ys[1], ys[2], ys[3] = Dot4BF16(x[t*in:][:in], w0, w1, w2, w3)
			}
		}
	}
}
