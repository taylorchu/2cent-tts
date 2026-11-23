package audio

import (
	"math"
	"strings"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// conv is a Conv1d. A depthwise one reads a single input channel per output channel, the one of the same index.
type conv struct {
	in, out, kernel, stride, padding, dilation int
	depthwise                                  bool
	weight                                     tensor.Values
	bias                                       []float32
}

// storedType is the type a release stores a codec or detector kernel in. The decoder's are bf16: rounded, they moved
// decoded audio far below the decoder's own noise. The encoder's and the quantizers' are f32: rounded, they changed
// several percent of a reference's codes. The detector's are f32 too, since it is small.
func storedType(name string) tensor.DType {
	if strings.HasPrefix(name, "snac.decoder.") {
		return tensor.BF16
	}
	return tensor.F32
}

// loadConv reads an undilated Conv1d, with a bias, of the stride and padding its model gives it.
func loadConv(l *tensor.Loader, prefix string, stride, padding int) conv {
	return readConv(l, prefix, l.Dims(prefix+".weight", 3), stride, padding, 1, true)
}

// loadSameConv reads an undilated Conv1d of stride one, padded so its output is as long as its input.
func loadSameConv(l *tensor.Loader, prefix string, bias bool) conv {
	shape := l.Dims(prefix+".weight", 3)
	return readConv(l, prefix, shape, 1, (shape[2]-1)/2, 1, bias)
}

// loadDepthwiseConv reads a same Conv1d whose every output reads only the input channel of its own index, so its
// stored input size is one.
func loadDepthwiseConv(l *tensor.Loader, prefix string, dilation int) conv {
	shape := l.Dims(prefix+".weight", 3)
	c := readConv(l, prefix, []int{shape[0], 1, shape[2]}, 1, (shape[2]-1)*dilation/2, dilation, true)
	c.depthwise, c.in = true, c.out
	return c
}

// readConv reads a Conv1d of shape out by in by kernel with its weight in the type the release stores it in.
func readConv(l *tensor.Loader, prefix string, shape []int, stride, padding, dilation int, bias bool) conv {
	c := conv{out: shape[0], in: shape[1], kernel: shape[2], stride: stride, padding: padding, dilation: dilation,
		weight: l.Values(prefix+".weight", storedType(prefix), shape...)}
	if bias {
		c.bias = l.Float32s(prefix+".bias", c.out)
	}
	return c
}

func (c conv) apply(x signal, sc *scratch) signal {
	positions := (x.positions+2*c.padding-c.dilation*(c.kernel-1)-1)/c.stride + 1
	y := sc.signal(c.out, positions)
	// dense writes four outputs at a time from the inputs they share; outputs that each read an input of their own,
	// or fewer than four, would leave most of each block idle, so each output adds its taps instead. Each tap adds a
	// shifted copy of an input channel; positions the shift moves outside the input add nothing, as zero padding would.
	inputs := c.in
	if c.depthwise {
		inputs = 1
	}

	if c.stride == 1 && (inputs == 1 || c.out < 4) {
		tiles := (positions + spreadTile - 1) / spreadTile
		tensor.RunParallel(&sc.spread, spreadTask{c: c, x: x, y: y, inputs: inputs, tiles: tiles}, c.out*tiles,
			inputs*spreadTile*c.kernel)
		return y
	}

	c.dense(y, x, sc)
	return y
}

// A spread work item is one output channel over a tile of spreadTile positions, so a single output still spreads over
// the cores.
const spreadTile = 1 << 13

type spreadTask struct {
	c             conv
	x, y          signal
	inputs, tiles int
}

func (task *spreadTask) Run(lo, hi, _ int) {
	c, x, y := task.c, task.x, task.y
	for item := lo; item < hi; item++ {
		o, t0 := item/task.tiles, item%task.tiles*spreadTile
		t1 := min(t0+spreadTile, y.positions)
		ys := y.row(o)
		// The taps add into the output, so it starts at the bias; the dense path writes each output once instead and
		// adds the bias there.
		for t := t0; t < t1; t++ {
			ys[t] = c.biasOf(o)
		}

		for i := range task.inputs {
			xs := x.row(i)
			if c.depthwise {
				xs = x.row(o)
			}

			for k := range c.kernel {
				wk := c.weight.At((o*task.inputs+i)*c.kernel + k)
				shift := k*c.dilation - c.padding
				first, last := max(t0, -shift), min(t1, len(xs)-shift)
				if first < last {
					tensor.Axpy(ys[first:last], wk, xs[first+shift:last+shift])
				}
			}
		}
	}
}

// dense sets y to the convolution of x, every output reading every input channel. A column is one input channel at one
// tap. Each worker copies a tile of positions into groups of eight contiguous positions, so a 4 by 8 block of outputs
// builds up in registers and the input is never copied whole, once per tap.
func (c conv) dense(y, x signal, sc *scratch) {
	tensor.SizedPerWorker(&sc.tiles, c.in*c.kernel*denseTilePositions)
	tensor.SizedPerWorker(&sc.rows, 4*c.in*c.kernel)
	blocks := (c.out + denseBlockOutputs - 1) / denseBlockOutputs
	tiles := (y.positions + denseTilePositions - 1) / denseTilePositions
	tensor.RunParallel(&sc.dense, denseTask{c: c, x: x, y: y, tiles: sc.tiles, rows: sc.rows, blocks: blocks},
		tiles*blocks, denseTilePositions*denseBlockOutputs*c.in*c.kernel)
}

// A dense work item is denseBlockOutputs outputs over denseTilePositions positions.
const denseTilePositions, denseBlockOutputs = 64, 64

type denseTask struct {
	c      conv
	x, y   signal
	tiles  [][]float32
	rows   [][]float32
	blocks int
}

func (task *denseTask) Run(lo, hi, worker int) {
	c, x, y := task.c, task.x, task.y
	packed, rows := task.tiles[worker][:c.in*c.kernel*denseTilePositions], task.rows[worker][:4*c.in*c.kernel]
	for item := lo; item < hi; item++ {
		t0 := item / task.blocks * denseTilePositions
		t1 := min(t0+denseTilePositions, y.positions)
		o0 := item % task.blocks * denseBlockOutputs
		o1 := min(o0+denseBlockOutputs, c.out)
		// A last group short of eight positions is filled out with the positions after the tile; each position's
		// outputs depend on that position alone, and only the tile's are copied out.
		groups := (t1 - t0 + 7) / 8
		c.packTile(packed, x, t0, groups)
		c.writeTile(y, packed, rows, t0, t1, o0, o1)
	}
}

// packTile copies groups of eight positions from t0 on into packed, column by column, each group's positions side by
// side.
func (c conv) packTile(packed []float32, x signal, t0, groups int) {
	cols := c.in * c.kernel
	for ch := range c.in {
		xs := x.row(ch)
		for k := range c.kernel {
			col := ch*c.kernel + k
			for g := range groups {
				group := packed[(g*cols+col)*8:][:8]
				at := (t0+8*g)*c.stride + k*c.dilation - c.padding
				// Most groups lie inside the input and, unstrided, side by side in it.
				if c.stride == 1 && at >= 0 && at+8 <= len(xs) {
					copy(group, xs[at:])
					continue
				}

				for j := range group {
					group[j] = 0
					if p := at + j*c.stride; p >= 0 && p < len(xs) {
						group[j] = xs[p]
					}
				}
			}
		}
	}
}

// writeTile sets outputs o0 to o1 at positions t0 to t1 of y from a tile packTile packed, the weights of four outputs
// at a time in f32 in rows.
func (c conv) writeTile(y signal, packed, rows []float32, t0, t1, o0, o1 int) {
	cols := c.in * c.kernel
	groups := (t1 - t0 + 7) / 8
	o := o0
	for ; o+4 <= o1; o += 4 {
		w := c.weight.Slice(o*cols, (o+4)*cols).Float32(rows)
		w0, w1, w2, w3 := w[:cols], w[cols:2*cols], w[2*cols:3*cols], w[3*cols:]
		for g := range groups {
			block := tensor.Kernel4x8(packed[g*cols*8:][:cols*8], w0, w1, w2, w3)
			for r := range 4 {
				for j, t := 0, t0+8*g; j < 8 && t < t1; j, t = j+1, t+1 {
					y.row(o + r)[t] = block[8*r+j] + c.biasOf(o+r)
				}
			}
		}
	}

	// The outputs left after the last group of four.
	for ; o < o1; o++ {
		w := c.weight.Slice(o*cols, (o+1)*cols).Float32(rows)
		for t := t0; t < t1; t++ {
			sum := c.biasOf(o)
			for col, wi := range w {
				sum += wi * packed[((t-t0)/8*cols+col)*8+(t-t0)%8]
			}
			y.row(o)[t] = sum
		}
	}
}

func (c conv) biasOf(o int) float32 {
	if c.bias == nil {
		return 0
	}
	return c.bias[o]
}

// transposedConv is a ConvTranspose1d.
type transposedConv struct {
	in, out, kernel, stride, padding int
	// taps holds W[:, :, k] transposed, out rows by in columns, for each tap k in turn.
	taps tensor.Values
	bias []float32
}

// loadTransposedConv reads a ConvTranspose1d, with a bias, of the stride and padding its model gives it.
func loadTransposedConv(l *tensor.Loader, prefix string, stride, padding int) transposedConv {
	shape := l.Dims(prefix+".weight", 3)
	in, out, kernel := shape[0], shape[1], shape[2]
	return transposedConv{in: in, out: out, kernel: kernel, stride: stride, padding: padding,
		taps: l.Values(prefix+".weight", storedType(prefix), shape...).Gathered(func(j int) int {
			i, o, k := j%in, j/in%out, j/(in*out)
			return (i*out+o)*kernel + k
		}),
		bias: l.Float32s(prefix+".bias", out)}
}

func (c transposedConv) apply(x signal, sc *scratch) signal {
	positions := (x.positions-1)*c.stride - 2*c.padding + c.kernel
	y := sc.signal(c.out, positions)
	// The taps add into the output, so it starts at the bias.
	if c.bias != nil {
		for o := range c.out {
			ys := y.row(o)
			for i := range ys {
				ys[i] = c.bias[o]
			}
		}
	}

	// One tap at a time, so only one tap's products are held in memory.
	tap := sc.signal(c.out, x.positions)
	for k := range c.kernel {
		weight := c.taps.Slice(k*c.out*c.in, (k+1)*c.out*c.in)
		conv{in: c.in, out: c.out, kernel: 1, stride: 1, dilation: 1, weight: weight}.dense(tap, x, sc)
		tensor.RunParallel(&sc.taps, tapTask{c: c, y: y, tap: tap, k: k}, c.out, x.positions)
	}
	sc.release(tap)
	return y
}

// tapTask adds one tap's products into the output, each at its place in the upsampled positions.
type tapTask struct {
	c      transposedConv
	y, tap signal
	k      int
}

func (task *tapTask) Run(lo, hi, _ int) {
	c := task.c
	for o := lo; o < hi; o++ {
		ys := task.y.row(o)
		for t, v := range task.tap.row(o) {
			at := t*c.stride - c.padding + task.k
			if at >= 0 && at < len(ys) {
				ys[at] += v
			}
		}
	}
}

type snake struct {
	alpha []float32
}

// loadSnake reads a snake's alpha, named prefix+".alpha", for a signal of channels channels.
func loadSnake(l *tensor.Loader, prefix string, channels int) snake {
	return snake{l.Float32s(prefix+".alpha", 1, channels, 1)}
}

// apply runs on the workers, alone of the elementwise stages: its sine costs as much as a small convolution's position,
// where relu, tanh and the noise cost too little to be worth splitting.
func (s snake) apply(x signal, sc *scratch) signal {
	y := sc.signal(x.channels, x.positions)
	const sineCost = 8
	tensor.RunParallel(&sc.snakes, snakeTask{s: s, x: x, y: y}, x.channels, x.positions*sineCost)
	return y
}

type snakeTask struct {
	s    snake
	x, y signal
}

func (task *snakeTask) Run(lo, hi, _ int) {
	for c := lo; c < hi; c++ {
		tensor.SnakeRow(task.y.row(c), task.x.row(c), task.s.alpha[c])
	}
}

type relu struct{}

func (relu) apply(x signal, _ *scratch) signal {
	for i, v := range x.data {
		x.data[i] = max(v, 0)
	}
	return x
}

type tanh struct{}

func (tanh) apply(x signal, _ *scratch) signal {
	for i, v := range x.data {
		x.data[i] = float32(math.Tanh(float64(v)))
	}
	return x
}

// residual adds its body's output to its input. Every body starts with snake, which writes a new signal, so the
// input is intact for the sum; the sequence holding the residual releases the input after it.
type residual struct {
	body sequence
}

func loadResidual(l *tensor.Loader, prefix string, dilation int) residual {
	depthwise := loadDepthwiseConv(l, prefix+".block.1", dilation)
	mix := loadSameConv(l, prefix+".block.3", true)
	return residual{sequence{loadSnake(l, prefix+".block.0", depthwise.in), depthwise,
		loadSnake(l, prefix+".block.2", mix.in), mix}}
}

func (r residual) apply(x signal, sc *scratch) signal {
	y := r.body.apply(x, sc)
	tensor.Add(y.data, x.data)
	return y
}

// noise adds Gaussian noise, one draw per position from sc.noise, scaled per channel by a learned projection of the
// signal.
type noise struct {
	linear conv
}

func (n noise) apply(x signal, sc *scratch) signal {
	h := n.linear.apply(x, sc)
	draws := tensor.Sized(&sc.draws, x.positions)
	for t := range draws {
		draws[t] = sc.noise(t)
	}

	for ch := range x.channels {
		xs, hs := x.row(ch), h.row(ch)
		for t := range xs {
			xs[t] += draws[t] * hs[t]
		}
	}
	sc.release(h)
	return x
}
