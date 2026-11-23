package audio

import (
	"fmt"
	"math"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// quantizer is one SNAC level: a codebook searched by direction, with a code every stride positions of the signal.
type quantizer struct {
	stride          int
	inProj, outProj conv
	// normalizedBook holds the codebook's vectors at unit length.
	codebook, normalizedBook []float32
}

// addEmbedded adds scale times the codes' codebook vectors, projected back to the signal's channels, to x, each code
// spanning stride positions of x.
func (q quantizer) addEmbedded(x signal, codes []int, scale float32, sc *scratch) {
	dimensions := q.inProj.out
	e := sc.signal(dimensions, len(codes))
	for t, c := range codes {
		for j := range dimensions {
			e.row(j)[t] = q.codebook[c*dimensions+j]
		}
	}

	y := q.outProj.apply(e, sc)
	sc.release(e)
	for ch := range x.channels {
		xs, ys := x.row(ch), y.row(ch)
		for t := range xs {
			xs[t] += scale * ys[t/q.stride]
		}
	}
	sc.release(y)
}

// nearestCodes is, for each position of e, the code whose vector points the nearest way. A reference is encoded once
// and its positions are few, so the search runs on one core.
func (q quantizer) nearestCodes(e signal) []int {
	dimensions := e.channels
	codes := make([]int, e.positions)
	vector := make([]float32, dimensions)
	for t := range e.positions {
		for j := range dimensions {
			vector[j] = e.row(j)[t]
		}

		best, bestDot := 0, float32(math.Inf(-1))
		for c := range LevelSize {
			if d := tensor.Dot(vector, q.normalizedBook[c*dimensions:][:dimensions]); d > bestDot {
				best, bestDot = c, d
			}
		}
		codes[t] = best
	}
	return codes
}

// SNAC is the 24 kHz SNAC codec, with three levels of codes, each at twice the rate of the one before. Its encoder
// and decoder are each a list of stages in the order of SNAC's own layers. It is only read once loaded, so any number
// of goroutines may share it; each decoding goroutine has a Stream of its own.
type SNAC struct {
	encoder    sequence
	quantizers []quantizer
	decoder    sequence
	// slotLevels is the SNAC level of each of a frame's codes, in the order training packed them.
	slotLevels []int
}

// CheckSlotLevels refuses slot levels that are not one SNAC 24 kHz frame's: one code at level 0, two at level 1 and
// four at level 2, in any order.
func CheckSlotLevels(levels []int) error {
	var perLevel [LevelCount]int
	for _, level := range levels {
		if level < 0 || level >= LevelCount {
			return fmt.Errorf("slot level %d is not a SNAC level", level)
		}
		perLevel[level]++
	}

	if perLevel != [LevelCount]int{1, 2, 4} {
		return fmt.Errorf("slot levels %v do not hold one frame", levels)
	}
	return nil
}

// LoadSNAC reads the codec's weights, named snac.*, for codes packed in slotLevels. A missing or misshapen tensor is
// the loader's error, which LoadSNAC returns too.
func LoadSNAC(l *tensor.Loader, slotLevels []int) (*SNAC, error) {
	err := CheckSlotLevels(slotLevels)
	if err != nil {
		return nil, err
	}

	s := &SNAC{slotLevels: slotLevels, encoder: sequence{loadSameConv(l, "snac.encoder.block.0", true)}}
	for i, stride := range []int{2, 4, 8, 8} {
		prefix := fmt.Sprintf("snac.encoder.block.%d.block", i+1)
		for j, dilation := range []int{1, 3, 9} {
			s.encoder = append(s.encoder, loadResidual(l, fmt.Sprintf("%s.%d", prefix, j), dilation))
		}
		down := loadConv(l, prefix+".4", stride, (stride+1)/2)
		s.encoder = append(s.encoder, loadSnake(l, prefix+".3", down.in), down)
	}
	s.encoder = append(s.encoder, loadDepthwiseConv(l, "snac.encoder.block.5", 1))

	for i, stride := range []int{4, 2, 1} {
		prefix := fmt.Sprintf("snac.quantizer.quantizers.%d", i)
		q := quantizer{stride: stride, inProj: loadSameConv(l, prefix+".in_proj", true),
			outProj: loadSameConv(l, prefix+".out_proj", true)}
		// A release keeps the codebooks in f32: rounded to bf16, they changed several percent of a reference's codes.
		q.codebook = l.Float32s(prefix+".codebook.weight", LevelSize, q.inProj.out)
		q.normalizedBook = make([]float32, len(q.codebook))
		// A missing projection reads as zero channels, which the loader reports; the loop must still end.
		for j := 0; q.inProj.out > 0 && j+q.inProj.out <= len(q.codebook); j += q.inProj.out {
			normalized := q.normalizedBook[j : j+q.inProj.out]
			normalize(normalized, q.codebook[j:j+q.inProj.out])
		}
		s.quantizers = append(s.quantizers, q)
	}

	s.decoder = sequence{loadDepthwiseConv(l, "snac.decoder.model.0", 1),
		loadSameConv(l, "snac.decoder.model.1", true)}
	for i, stride := range []int{8, 8, 4, 2} {
		prefix := fmt.Sprintf("snac.decoder.model.%d.block", i+2)
		up := loadTransposedConv(l, prefix+".1", stride, (stride+1)/2)
		s.decoder = append(s.decoder, loadSnake(l, prefix+".0", up.in), up,
			noise{linear: loadSameConv(l, prefix+".2.linear", false)})
		for j, dilation := range []int{1, 3, 9} {
			s.decoder = append(s.decoder, loadResidual(l, fmt.Sprintf("%s.%d", prefix, j+3), dilation))
		}
	}
	last := loadSameConv(l, "snac.decoder.model.7", true)
	s.decoder = append(s.decoder, loadSnake(l, "snac.decoder.model.6", last.in), last, tanh{})

	if l.Err() != nil {
		return nil, l.Err()
	}

	return s, nil
}

// normalize writes x / max(|x|, 1e-12), as torch's F.normalize does.
func normalize(out, x []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}

	norm := float32(max(math.Sqrt(sum), 1e-12))
	for i, v := range x {
		out[i] = v / norm
	}
}

// decode turns frame codes into audio in sc's memory, with sc's noise: the audio is sc's, and the next decode in sc
// reuses it.
func (s *SNAC) decode(codes []int, sc *scratch) []float32 {
	frames := len(codes) / FrameCodes
	for level := range sc.levels {
		sc.levels[level] = sc.levels[level][:0]
	}

	for f := range frames {
		for slot, level := range s.slotLevels {
			sc.levels[level] = append(sc.levels[level], codes[f*FrameCodes+slot]%LevelSize)
		}
	}

	first := s.quantizers[0]
	z := sc.signal(first.outProj.out, frames*first.stride)
	for level, q := range s.quantizers {
		q.addEmbedded(z, sc.levels[level], 1, sc)
	}

	out := s.decoder.apply(z, sc)
	sc.release(z)
	// The audio stays sc's until the next decode, which takes it back first.
	sc.release(out)
	return out.row(0)
}

const (
	// decodeContext is how many frames on each side of a span shape its audio: decoded with that many, a span is
	// sample for sample the same span of a whole decode, except for the decoder's noise, which each span draws afresh.
	decodeContext = 3
	// decodeSpan frames are decoded and sent at a time: about a second, so speech starts soon and the decoder's
	// signals stay small, at the cost of decoding each span's context again.
	decodeSpan = 12
)

// Stream decodes speech frame by frame, a span at a time, span after span in the same memory. Its speech runs on until
// Flush.
type Stream struct {
	snac    *SNAC
	scratch *scratch
	codes   []int
	// sent is how many of the speech's frames have been returned as audio.
	sent int
}

// NewStream is a stream that decodes with s, noise giving the decoder's Gaussian noise, one value per position of the
// signal it adds to. A Stream serves one goroutine at a time; the SNAC it decodes with may serve any number.
func (s *SNAC) NewStream(noise func(position int) float32) *Stream {
	return &Stream{snac: s, scratch: &scratch{noise: noise}}
}

// Reset starts the stream over, keeping its memory for later speech.
func (s *Stream) Reset() {
	s.codes, s.sent = s.codes[:0], 0
}

// Add takes the speech's next frame and returns the audio that became final with it, if any. The audio is the
// stream's, read until the next Add or Flush.
func (s *Stream) Add(frame []int) []float32 {
	if len(frame) != FrameCodes {
		panic(fmt.Sprintf("a frame of %d codes, want %d", len(frame), FrameCodes))
	}

	s.codes = append(s.codes, frame...)
	end := s.sent + decodeSpan
	if len(s.codes)/FrameCodes < end+decodeContext {
		return nil
	}
	return s.decodeTo(end)
}

// Flush returns the audio of the speech's frames not yet returned, if any, and starts the stream over for the next
// speech. The audio is the stream's, read until the next Add or Flush.
func (s *Stream) Flush() []float32 {
	audio := s.decodeTo(len(s.codes) / FrameCodes)
	s.Reset()
	return audio
}

// decodeTo returns the audio of the frames from sent to end, decoded with the context around them that it has.
func (s *Stream) decodeTo(end int) []float32 {
	var audio []float32
	if end > s.sent {
		first, last := max(0, s.sent-decodeContext), min(len(s.codes)/FrameCodes, end+decodeContext)
		audio = s.snac.decode(s.codes[first*FrameCodes:last*FrameCodes], s.scratch)
		audio = audio[(s.sent-first)*FrameSamples : (end-first)*FrameSamples]
	}

	s.sent = end
	// Frames older than the context of the next span are never decoded again, so speech carried on for a whole script
	// keeps a stream's memory to a span and its context.
	if old := s.sent - decodeContext; old > 0 {
		s.codes = s.codes[:copy(s.codes, s.codes[old*FrameCodes:])]
		s.sent -= old
	}
	return audio
}

// Encode turns whole frames of 24 kHz audio into frame codes in the model's layout, each level in its own range. A
// partial frame panics, as a frame of the wrong length does in Add: a level would encode codes past the frames the
// others hold. Its signals at the audio's full rate are large and an encode is rare, so they are not kept.
func (s *SNAC) Encode(samples []float32) []int {
	if len(samples)%FrameSamples != 0 {
		panic(fmt.Sprintf("%d samples are not whole frames of %d", len(samples), FrameSamples))
	}

	sc := new(scratch)
	// Each level quantizes the remainder of the encoding the levels before it leave.
	remainder := s.encoder.apply(signal{1, len(samples), samples}, sc)
	var levels [LevelCount][]int
	for level, q := range s.quantizers {
		pooled := averagePool(remainder, q.stride, sc)
		projected := q.inProj.apply(pooled, sc)
		sc.release(pooled)
		levels[level] = q.nearestCodes(projected)
		sc.release(projected)
		q.addEmbedded(remainder, levels[level], -1, sc)
	}
	sc.release(remainder)
	return s.frameLayout(levels)
}

// averagePool is x averaged over each run of stride positions.
func averagePool(x signal, stride int, sc *scratch) signal {
	pooled := sc.signal(x.channels, x.positions/stride)
	for ch := range x.channels {
		xs, ps := x.row(ch), pooled.row(ch)
		for t := range ps {
			var sum float32
			for _, v := range xs[t*stride : (t+1)*stride] {
				sum += v
			}
			ps[t] = sum / float32(stride)
		}
	}
	return pooled
}

// frameLayout interleaves the levels' codes a frame at a time in the slot order the model reads, each level in its own
// range: the layout decode takes apart.
func (s *SNAC) frameLayout(levels [LevelCount][]int) []int {
	var codes []int
	var next [LevelCount]int
	for range levels[0] {
		for _, level := range s.slotLevels {
			codes = append(codes, levels[level][next[level]]+level*LevelSize)
			next[level]++
		}
	}
	return codes
}
