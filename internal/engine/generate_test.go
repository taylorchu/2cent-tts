package engine

import (
	"errors"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
	"github.com/taylorchu/2cent-tts/internal/model"
)

// Two Go passes that read the same positions run the same kernels in the same order whatever their batches, so they
// agree to the bit; this leaves room only for a future kernel that sums a batch's rows in another order.
const sameTolerance = 1e-6

// The golden scores come from training's pass, whose attention holds keys and values unrounded, while the engine's
// caches round them to bf16, so a score moves by a few hundredths whatever its size. What a draw depends on is its
// distribution: this bounds the chance, at the sampling temperature of one, that a draw comes out otherwise. A
// decision's logit is held to the same bound on the distribution training learned of whether a unit started.
const drawTolerance = 3e-2

// The engine starts a unit where a decision's logit is above zero. Rounding moves a logit by a few hundredths, so one
// this far from zero must keep its sign.
const startMargin = 0.1

// fixtureVoice is the voice and the chunk of the golden generation, built as Speak builds them, after a check that
// their prompt is the row training reads, cut where its target opens.
func fixtureVoice(tb testing.TB, e *Engine, g *fixtures.Golden) (*Voice, chunk) {
	tb.Helper()
	prefix, err := voicePrompt(e, g.Prompts.ReferenceText, g.Reference.Codes)
	if err != nil {
		tb.Fatal(err)
	}

	golden := chunk{units: g.Model.Units, textUnits: len(g.Model.Units) - e.settings.LookaheadUnits}
	open, _ := golden.split(e.settings.LookaheadUnits)
	prompt := e.markers.appendOpening(e.markers.appendChunkStatic(slices.Clone(prefix), golden), open)
	if !slices.Equal(prompt, g.Model.Prompt) {
		tb.Fatalf("the prompt holds %d tokens, want the golden %d", len(prompt), len(g.Model.Prompt))
	}

	unconditional := e.markers.appendOpening(e.markers.unconditionalPrompt(), nil)
	if !slices.Equal(unconditional, g.Model.Unconditional) {
		tb.Fatal("the unconditional prompt is not the golden one")
	}

	return e.readVoice(new(model.State), prefix), golden
}

// clone is a copy of cache that shares no memory with it.
func clone(cache *model.Cache) *model.Cache {
	out := new(model.Cache)
	out.CopyFrom(cache, 0)
	return out
}

// TestPassesBatched reads a step with both passes as one batch and with each pass alone: the vectors and the head's
// scores must match bit for bit, since a batch only shares the reads of the weights.
func TestPassesBatched(t *testing.T) {
	e, g := releaseEngine(t)
	v, _ := fixtureVoice(t, e, g)
	m, st := e.model, new(model.State)
	input := m.StepInput(st, make([]int, model.StepCodes))
	prefixes := []*model.Cache{v.prompt, e.unconditional}
	h := m.MainHidden()
	vectors := [][]float32{make([]float32, h), make([]float32, h)}
	m.ReadMainStep(st, []*model.Cache{clone(prefixes[0]), clone(prefixes[1])}, vectors, input)
	for i, prefix := range prefixes {
		alone := make([]float32, h)
		m.ReadMain(st, clone(prefix), alone, [][]float32{input})
		if !slices.Equal(alone, vectors[i]) {
			t.Errorf("pass %d: the main model's vector read in a batch differs from the one read alone", i)
		}
	}

	// The head's scores, every slot of a step, read in one batch and for each pass alone. A later slot attends to the
	// earlier ones, so the heads' caches are read too.
	written := func(heads []*model.Cache, vectors [][]float32) [][][]float32 {
		var scores [][][]float32
		m.WriteStep(st, heads, vectors, func(s [][]float32) int {
			// The scores are st's until the next slot.
			var kept [][]float32
			for _, row := range s {
				kept = append(kept, slices.Clone(row))
			}
			scores = append(scores, kept)
			// Each slot picks another code, so a later slot reads earlier codes that differ.
			return len(scores)
		})
		return scores
	}

	batched := written([]*model.Cache{m.NewHeadCache(), m.NewHeadCache()}, vectors)
	for i := range vectors {
		alone := written([]*model.Cache{m.NewHeadCache()}, vectors[i:i+1])
		for slot := range alone {
			if !slices.Equal(alone[slot][0], batched[slot][i]) {
				t.Errorf("pass %d slot %d: the head's scores in a batch differ from the ones alone", i, slot)
			}
		}
	}
}

// TestWarmChunkAllocatesNothing speaks the golden chunk after the tail of its own speech, decoding every step, in a
// request that spoke it before: the model, the fed tail and the decoder all work in the request's memory. Packing a
// chunk and filling its tail allocate on purpose and are left out. The tail it feeds runs past the window, so a few
// written steps after it are enough.
func TestWarmChunkAllocatesNothing(t *testing.T) {
	release, g := releaseEngine(t)
	v, golden := fixtureVoice(t, release, g)
	carried := goldenSpeech(t, release, g, v, golden).tail
	e, _ := withSettings(t, func(s *settings) { s.MaxSeconds = stepSeconds(10) })
	spoken := carried.prependTo(golden)
	r, source := e.takeRequest(), rand.NewPCG(3, 4)
	// Each run draws the same speech, so a run allocates nothing only if every run fits the memory the first sized.
	speak := func() {
		source.Seed(3, 4)
		r.seed(source)
		r.decoder.Reset()
		err := r.generate(t.Context(), v, spoken, carried, func(codes []int) error {
			return r.decode(codes, func([]float32) error { return nil })
		})
		if err != nil && !errors.Is(err, errCapped) {
			t.Fatal(err)
		}

		r.decoder.Flush()
	}

	if n := fixtures.Allocations(1, speak); n != 0 {
		t.Errorf("a warm request allocated %d times for a chunk", n)
	}
}

// TestGenerate steers generation onto the golden codes and starts and checks every draw's guided distribution and
// every decision against what training reads in one pass over the same rows.
func TestGenerate(t *testing.T) {
	e, g := releaseEngine(t)
	v, golden := fixtureVoice(t, e, g)
	run := goldenSpeech(t, e, g, v, golden)
	if len(run.scores) != len(g.Model.Guided) || len(run.decisions) != len(g.Model.Decisions) {
		t.Fatalf("%d draws and %d decisions, want %d and %d", len(run.scores), len(run.decisions),
			len(g.Model.Guided), len(g.Model.Decisions))
	}

	for i, scores := range run.scores {
		if d := totalVariation(scores, fixtures.Floats(t, g.Model.Guided[i])); d > drawTolerance {
			t.Fatalf("draw %d: its distribution is %.4f from the golden one, more than %.4f", i, d, drawTolerance)
		}
	}

	for i, logit := range run.decisions {
		w := float32(g.Model.Decisions[i])
		if d := totalVariation([]float32{logit, 0}, []float32{w, 0}); d > drawTolerance {
			t.Fatalf("step %d: decision %f is %.4f from the golden %f, more than %.4f", i, logit, d, w, drawTolerance)
		}

		if math.Abs(float64(w)) > startMargin && (logit > 0) != (w > 0) {
			t.Fatalf("step %d: decision %f starts a unit otherwise than the golden %f", i, logit, w)
		}
	}

	// The speech must run past the window, or the cut that keeps it is not checked.
	if steps := len(g.Model.Starts); steps <= e.settings.MainWindow {
		t.Errorf("the golden speech holds %d steps, within the window of %d", steps, e.settings.MainWindow)
	}
}

// TestGenerateReportsTheCap speaks with no start ever said, so the speech runs to the cap, which generate must report.
func TestGenerateReportsTheCap(t *testing.T) {
	e, g := withSettings(t, func(s *settings) { s.MaxSeconds = stepSeconds(3) })
	v, golden := fixtureVoice(t, e, g)
	r := e.takeRequest()
	r.choose = chooser{draw: func([]float32) int { return 0 }, started: func(float32) bool { return false }}
	steps := 0
	err := r.generate(t.Context(), v, golden, nil, func([]int) error {
		steps++
		return nil
	})
	if !errors.Is(err, errCapped) || steps != 3 {
		t.Errorf("%d steps and error %v, want 3 and the cap", steps, err)
	}
}

// TestSampleLowTemperaturePicksTheLargest draws at a temperature near zero, where only the largest logit is drawn.
func TestSampleLowTemperaturePicksTheLargest(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	logits := make([]float32, 64)
	for range 200 {
		for i := range logits {
			logits[i] = float32(rng.NormFloat64())
		}

		got := sample(logits, 1e-3, rng, make([]float32, len(logits)))
		if want := slices.Index(logits, slices.Max(logits)); got != want {
			t.Fatalf("drew %d, want the largest logit's %d", got, want)
		}
	}
}

// TestSampleFollowsTheSoftmax draws many times and compares each code's share with the softmax of the logits divided
// by the temperature.
func TestSampleFollowsTheSoftmax(t *testing.T) {
	const temperature = 0.7
	logits := []float32{0, 1, 2, -1, 0.5}
	rng := rand.New(rand.NewPCG(9, 10))
	checkShares(t, 200000, logits, temperature, func() int {
		return sample(logits, temperature, rng, make([]float32, len(logits)))
	})
}

// checkShares draws n codes with draw and compares each code's share with the softmax of scores divided by
// temperature.
func checkShares(t *testing.T, n int, scores []float32, temperature float64, draw func() int) {
	t.Helper()
	counts := make([]int, len(scores))
	for range n {
		counts[draw()]++
	}

	scaled := make([]float32, len(scores))
	for i, s := range scores {
		scaled[i] = float32(float64(s) / temperature)
	}

	for i, want := range probabilities(scaled) {
		if share := float64(counts[i]) / float64(n); !approx.Equal(share, want, 0.005, 0) {
			t.Errorf("code %d drawn %.4f of the time, want %.4f at temperature %g", i, share, want, temperature)
		}
	}
}

// TestSampleSkipsImpossibleCodes draws zero, the lowest value a generator gives, where a code whose probability
// underflowed to zero comes first.
func TestSampleSkipsImpossibleCodes(t *testing.T) {
	if got := sample([]float32{-1000, 0}, 1, rand.New(zeroSource{}), make([]float32, 2)); got != 1 {
		t.Errorf("drew code %d, whose probability is zero", got)
	}
}

type zeroSource struct{}

func (zeroSource) Uint64() uint64 {
	return 0
}

// TestDrawUsesTheReleaseTemperature draws many codes with the request's own chooser at a temperature far from one and
// compares each code's share with the softmax of the scores divided by the release's temperature.
func TestDrawUsesTheReleaseTemperature(t *testing.T) {
	e, _ := withSettings(t, func(s *settings) { s.Temperature = 0.4 })
	r := e.takeRequest()
	r.seed(rand.NewPCG(13, 14))
	scores := []float32{0, 1, 0.5, -1}
	checkShares(t, 100000, scores, e.settings.Temperature, func() int { return r.choose.draw(scores) })
}

// TestStartsAtTheTrainedBoundary decides with the request's own rule on either side of zero, the logit where the
// trained decision's probability is one half.
func TestStartsAtTheTrainedBoundary(t *testing.T) {
	e, _ := releaseEngine(t)
	r := e.takeRequest()
	if !r.choose.started(1e-3) || r.choose.started(-1e-3) {
		t.Error("the start rule does not cut at a logit of zero")
	}
}

// BenchmarkStep is one step of speech in a voice: both passes through the head for every slot of a step, the reader,
// and the main model reading the step back. Real time needs a step in less than the step's own duration.
func BenchmarkStep(b *testing.B) {
	// Each run speaks this many steps: the cap allows no more, and no start is said to end it sooner.
	const steps = 20
	e, g := withSettings(b, func(s *settings) { s.MaxSeconds = stepSeconds(steps) })
	v, golden := fixtureVoice(b, e, g)
	r := e.takeRequest()
	r.seed(rand.NewPCG(1, 2))
	r.choose.started = func(float32) bool { return false }
	for b.Loop() {
		_ = r.generate(b.Context(), v, golden, nil, func([]int) error { return nil })
	}
	b.ReportMetric(float64(b.Elapsed().Milliseconds())/float64(b.N*steps), "ms/step")
}

// totalVariation is how far apart the distributions two slots' scores give are: half the sum of the differences of
// each code's probability, or infinity when a score is NaN or the slots hold different numbers of codes.
func totalVariation(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	pa, pb := probabilities(a), probabilities(b)
	d := 0.0
	for i := range pa {
		d += math.Abs(pa[i] - pb[i])
	}

	if math.IsNaN(d) {
		return math.Inf(1)
	}
	return d / 2
}

func probabilities(scores []float32) []float64 {
	top := float64(slices.Max(scores))
	p, sum := make([]float64, len(scores)), 0.0
	for i, s := range scores {
		p[i] = math.Exp(float64(s) - top)
		sum += p[i]
	}

	for i := range p {
		p[i] /= sum
	}
	return p
}
