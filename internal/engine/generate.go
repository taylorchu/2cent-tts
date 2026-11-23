package engine

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// Guidance compares two passes of the main model: the one that reads the voice and the chunk's text, and the
// unconditional one, which reads neither. Every batch the two passes read or write holds them in this order.
const (
	withTextPass = iota
	unconditionalPass
)

// copyPrompt copies a read prompt into the cache of pass, kept from chunk to chunk, with room for extra positions past
// it and for what a read adds past the static prompt: at most the window and one position.
func (r *request) copyPrompt(pass int, prompt *model.Cache, extra int) {
	r.caches[pass].CopyFrom(prompt, extra+r.engine.settings.MainWindow+1)
}

// openPasses sets both passes up for a chunk: a copy of the voice's prompt that has read the chunk's whole text,
// which stays in view while the chunk is spoken, and a copy of the unconditional prompt, read once at load since it
// holds no chunk's text.
func (r *request) openPasses(v *Voice, c chunk) {
	e := r.engine
	r.ids = e.markers.appendChunkStatic(r.ids[:0], c)
	r.copyPrompt(withTextPass, v.prompt, len(r.ids))
	// Each position of a static prompt attends to all before it, as in training, so it is read in one call.
	e.model.ReadMain(&r.state, r.caches[withTextPass], nil, r.inputs(r.ids))
	r.static[withTextPass] = r.caches[withTextPass].Len()
	r.copyPrompt(unconditionalPass, e.unconditional, 0)
	r.static[unconditionalPass] = e.unconditional.Len()
}

// inputs is the main model's input for each position of ids, in the request's memory, which holds every input of a
// chunk until generate starts the next one.
func (r *request) inputs(ids []int) [][]float32 {
	start := len(r.inputRows)
	r.inputRows, r.inputValues = r.engine.model.AppendInputs(&r.state, r.inputRows, r.inputValues, ids)
	return r.inputRows[start:len(r.inputRows):len(r.inputRows)]
}

// read runs inputs through pass and copies the last one's vector into the pass's vector. It reads in batches small
// enough that one cut of the cache to the window gives each position of a batch just what it attended to in training.
func (r *request) read(pass int, inputs [][]float32) {
	cache, static, window := r.caches[pass], r.static[pass], r.engine.settings.MainWindow
	for len(inputs) > 0 {
		cache.KeepWindow(static, window)
		n := min(len(inputs), static+window-cache.Len()+1)
		r.engine.model.ReadMain(&r.state, cache, r.vectors[pass], inputs[:n])
		inputs = inputs[n:]
	}
}

// hear has both passes read a step, written or fed, each after its cache is cut to the window, and adds it to the
// speech heard. The passes share each read of the main model's weights.
func (r *request) hear(codes []int) {
	e := r.engine
	r.heard.add(codes)
	for pass, cache := range r.caches {
		cache.KeepWindow(r.static[pass], e.settings.MainWindow)
	}
	e.model.ReadMainStep(&r.state, r.caches[:], r.vectors[:], e.model.StepInput(&r.state, codes))
}

// startNext marks the last step heard as one a unit started in and has the pass with text read the next unit waiting.
// It reports false when no unit is waiting: the start that ends the speech.
func (r *request) startNext() bool {
	if r.fed == len(r.waiting) {
		return false
	}

	r.heard.started[len(r.heard.started)-1] = true
	r.read(withTextPass, r.waiting[r.fed])
	r.fed++
	return true
}

// chooser makes generation's two choices: the code a slot's guided scores give, and whether a decision's logit says
// a unit started.
type chooser struct {
	draw    func(scores []float32) int
	started func(logit float32) bool
}

// generate writes a chunk's speech in voice with r.choose and hands each step's codes to emitStep as it is written;
// emitStep's error ends the speech. It leaves every step the passes read, fed or written, with the starts decided in
// them, in r.heard, and returns errCapped for speech that MaxSeconds cut off.
//
// Guidance compares the pass with the voice and the text against the unconditional one. Both passes read every step,
// the carried tail's first, as if written. Only the pass with text reads the text: the first LookaheadUnits units when
// the target opens, then the next unit after each step a unit started in. The start that finds no unit left ends the
// speech.
//
// Each model call takes a turn of the cores and emitStep runs between turns, so requests at once take turns step by
// step and a slow step holds up no other request.
func (r *request) generate(ctx context.Context, voice *Voice, c chunk, carried *tail,
	emitStep func(codes []int) error) error {
	e := r.engine
	err := e.withTurn(ctx, func() { r.open(voice, c) })
	if err != nil {
		return err
	}

	err = r.feed(ctx, carried)
	if err != nil {
		return err
	}

	for range e.settings.maxSteps() {
		ended := false
		err := e.withTurn(ctx, func() { ended = r.writeStep() })
		if err != nil {
			return err
		}

		err = emitStep(r.heard.steps[len(r.heard.steps)-1])
		if err != nil {
			return err
		}

		if ended {
			return nil
		}
	}
	return errCapped
}

// open sets the request up to speak chunk c in voice: both passes read their static prompts and their target's
// opening, the units fed later wait as inputs, the heads hold nothing, and nothing is heard yet.
func (r *request) open(voice *Voice, c chunk) {
	e := r.engine
	r.heard.empty()
	r.inputRows, r.inputValues = r.inputRows[:0], r.inputValues[:0]
	r.openPasses(voice, c)
	opening, later := c.split(e.settings.LookaheadUnits)
	r.waiting, r.fed = r.waiting[:0], 0
	for _, unit := range later {
		r.waiting = append(r.waiting, r.inputs(unit))
	}

	tensor.SizedEach(r.vectors[:], e.model.MainHidden())
	r.ids = e.markers.appendOpening(r.ids[:0], opening)
	r.read(withTextPass, r.inputs(r.ids))
	r.read(unconditionalPass, e.unconditionalOpening)
	for _, cache := range r.headCaches {
		cache.Reset()
	}
}

// writeStep writes a step with r.choose, has both passes hear it and decides whether a unit started in it. It reports
// whether the speech ended: a start with no unit left to feed.
func (r *request) writeStep() bool {
	e := r.engine
	r.hear(e.model.WriteStep(&r.state, r.headCaches[:], r.vectors[:], r.pick))
	return r.choose.started(e.model.Decide(&r.state, r.vectors[withTextPass])) && !r.startNext()
}

// pick is WriteStep's pick for a written step: the chooser's draw from the two passes' guided scores. Every code of a
// step is drawn from thousands of scores, so the draws share the request's buffers.
func (r *request) pick(scores [][]float32) int {
	n := len(scores[withTextPass])
	return r.choose.draw(guided(tensor.Sized(&r.guided, n), tensor.Sized(&r.buffer, n), scores[withTextPass],
		scores[unconditionalPass], r.engine.settings.GuidanceScale))
}

// feed reads the steps of t into both passes as generate reads the ones it writes, a turn each, with the next unit
// waiting after each step a unit started in.
func (r *request) feed(ctx context.Context, t *tail) error {
	if t == nil {
		return nil
	}

	for at := range t.speech.steps {
		err := r.engine.withTurn(ctx, func() { r.feedStep(t, at) })
		if err != nil {
			return err
		}
	}
	return nil
}

// feedStep has the heads write step at of t with its known codes and both passes hear it. A head attends to its last
// steps alone, but what it kept of them depends on the steps before, layer by layer, so it writes every fed step.
func (r *request) feedStep(t *tail, at int) {
	codes := t.speech.steps[at]
	r.known.codes, r.known.slot = codes, 0
	r.engine.model.WriteStep(&r.state, r.headCaches[:], r.vectors[:], r.known.pick)
	r.hear(codes)
	if t.speech.started[at] {
		r.startNext()
	}
}

// knownCodes picks a step's codes, one each slot, whatever the scores. A step holds codes numbered across the
// levels, and a pick is a code's index within its level.
type knownCodes struct {
	codes []int
	slot  int
}

func (k *knownCodes) pick([][]float32) int {
	k.slot++
	return k.codes[k.slot-1] % audio.LevelSize
}

// errCapped is generate's report of speech that did not end within MaxSeconds; the speech it wrote stands.
var errCapped = errors.New("no end of speech within the cap")

// guided writes the guided log-probabilities into out, c + (scale - 1) * (c - u), c the pass with text's and u
// the unconditional one's. buffer is as long as withText.
func guided(out, buffer, withText, unconditional []float32, scale float64) []float32 {
	withTextNorm, unconditionalNorm := logSumExp(withText, buffer), logSumExp(unconditional, buffer)
	for i := range out {
		c, u := withText[i]-withTextNorm, unconditional[i]-unconditionalNorm
		out[i] = c + float32(scale-1)*(c-u)
	}
	return out
}

func logSumExp(x, buffer []float32) float32 {
	best := tensor.ExpFromMax(buffer, x, 1)
	return best + float32(math.Log(float64(tensor.Sum(buffer))))
}

// sample draws a code from logits at temperature, probabilities being a buffer as long as logits. The total and the
// running subtraction are float64, so a draw near the total still lands on a code.
func sample(logits []float32, temperature float64, rng *rand.Rand, probabilities []float32) int {
	tensor.ExpFromMax(probabilities, logits, float32(1/temperature))
	var total float64
	for _, p := range probabilities {
		total += float64(p)
	}

	u := rng.Float64() * total
	for i, p := range probabilities {
		u -= float64(p)
		if u < 0 {
			return i
		}
	}
	return len(probabilities) - 1
}
