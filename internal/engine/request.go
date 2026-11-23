package engine

import (
	"math/rand/v2"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// request is the memory one Speak works in, from its first chunk to its last: the model's state, the two passes and
// the heads' caches, the buffers of a code's draw, the chunk's prompt and inputs, the speech heard, the tail carried
// on, the decoder and the random generators. Nothing in it is shared with another request, so requests that run at
// once cannot touch each other's speech; the engine's turn only keeps them from sharing the cores.
type request struct {
	engine *Engine
	state  model.State
	// Each pass's main cache, the static prompt's length within it, its head's cache and its vector, indexed by
	// withTextPass and unconditionalPass. A position of the target attends to the static prompt and to the MainWindow
	// positions behind it; only the engine knows where a static prompt ends, so the engine cuts the window.
	caches                        [2]*model.Cache
	static                        [2]int
	headCaches                    [2]*model.Cache
	vectors                       [2][]float32
	guided, buffer, probabilities []float32
	// choose is the sampling chooser, made once so a chunk makes no closure; a test steers generation with its own.
	choose      chooser
	known       knownCodes
	ids         []int
	inputRows   [][]float32
	inputValues []float32
	// waiting holds the inputs of the units fed after the opening; fed counts the ones the pass with text has read.
	waiting [][][]float32
	fed     int
	heard   speech
	tail    tail
	decoder *audio.Stream
	// rng draws the codes from the caller's source. The decoder's noise has a generator of its own, seeded from rng,
	// so the audio of a seed does not depend on when the decoder runs.
	rng         rand.Rand
	noiseSource rand.PCG
	noise       rand.Rand
}

// newRequest is a request with its caches made and no other memory sized yet.
func (e *Engine) newRequest() *request {
	r := &request{engine: e}
	for pass := range r.caches {
		r.caches[pass], r.headCaches[pass] = e.model.NewMainCache(), e.model.NewHeadCache()
	}
	r.noise = *rand.New(&r.noiseSource)
	r.decoder = e.snac.NewStream(r.noiseAt)
	r.choose = chooser{
		draw: func(scores []float32) int {
			probabilities := tensor.Sized(&r.probabilities, len(scores))
			return sample(scores, e.settings.Temperature, &r.rng, probabilities)
		},
		// A logit above zero, the trained decision's boundary, is a start.
		started: func(logit float32) bool { return logit > 0 },
	}
	return r
}

// seed makes r draw from source, and its decoder's noise from a seed it draws first.
func (r *request) seed(source rand.Source) {
	r.rng = *rand.New(source)
	r.noiseSource.Seed(r.rng.Uint64(), r.rng.Uint64())
}

// noiseAt is the decoder's noise. Every position draws the next value of the request's noise generator, so the
// position itself is not needed.
func (r *request) noiseAt(int) float32 {
	return float32(r.noise.NormFloat64())
}

// takeRequest is a request in memory an earlier one left when there is one, so a server's requests allocate little.
func (e *Engine) takeRequest() *request {
	select {
	case r := <-e.requests:
		return r
	default:
		return e.newRequest()
	}
}

// giveBack keeps r's memory for a later request while fewer than Options.KeptRequests are kept.
func (e *Engine) giveBack(r *request) {
	select {
	case e.requests <- r:
	default:
	}
}
