package audio

import "unsafe"

// stage is one layer of the codec: it turns one signal into the next, and may overwrite its input to do so. A new
// signal comes from sc.
type stage interface {
	apply(x signal, sc *scratch) signal
}

type sequence []stage

// apply runs the stages in turn. Each signal between two stages goes back to sc once the next stage is done with it;
// x is the caller's, so it stays.
func (s sequence) apply(x signal, sc *scratch) signal {
	in := x
	for _, current := range s {
		y := current.apply(x, sc)
		if !y.same(x) && !x.same(in) {
			sc.release(x)
		}
		x = y
	}
	return x
}

// signal is channels by positions, one channel's positions contiguous.
type signal struct {
	channels, positions int
	data                []float32
}

func newSignal(channels, positions int) signal {
	return signal{channels, positions, make([]float32, channels*positions)}
}

func (s signal) row(c int) []float32 {
	return s.data[c*s.positions:][:s.positions]
}

// same is whether s and t are one signal's memory.
func (s signal) same(t signal) bool {
	return unsafe.SliceData(s.data) == unsafe.SliceData(t.data)
}

// scratch is the memory a decode or an encode works in. A signal no stage reads any more goes back to it and
// comes out again for a later stage or span, so a stream decodes span after span in the same memory. Each worker
// keeps a packed tile, and each kind of parallel work keeps the task that carries its arguments to the workers. One
// goroutine uses a scratch at a time; the zero value is ready to use.
type scratch struct {
	free  [][]float32
	tiles [][]float32
	rows  [][]float32
	draws []float32
	// noise gives the decoder's Gaussian noise, one value per position of the signal it adds to.
	noise  func(position int) float32
	levels [LevelCount][]int
	spread spreadTask
	dense  denseTask
	taps   tapTask
	snakes snakeTask
}

// signal is a zeroed signal, in memory a released one held when one was large enough. The smallest such is taken,
// so the large ones stay for the stages that need them.
func (sc *scratch) signal(channels, positions int) signal {
	n := channels * positions
	best := -1
	for i, buf := range sc.free {
		if cap(buf) >= n && (best < 0 || cap(buf) < cap(sc.free[best])) {
			best = i
		}
	}

	if best < 0 {
		return newSignal(channels, positions)
	}

	data := sc.free[best][:n]
	sc.free[best] = sc.free[len(sc.free)-1]
	sc.free = sc.free[:len(sc.free)-1]
	clear(data)
	return signal{channels, positions, data}
}

// release gives s back to sc. s may still be read until sc hands its memory out again.
func (sc *scratch) release(s signal) {
	if cap(s.data) > 0 {
		sc.free = append(sc.free, s.data[:0])
	}
}
