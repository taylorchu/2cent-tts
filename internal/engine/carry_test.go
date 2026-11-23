package engine

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/approx"
	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/fixtures"
	"github.com/taylorchu/2cent-tts/internal/model"
	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// TestCarriedTail picks a tail from speech whose units start at known steps: the latest start after which more steps
// than the window were spoken, however many tokens the units hold.
func TestCarriedTail(t *testing.T) {
	unit := func(tokens int) []int { return make([]int, tokens) }
	texts := []string{"a ", "b ", "c ", "d ", "e."}
	// Five units, the fourth two tokens long, then the two </s> of a chunk.
	c := chunk{units: [][]int{unit(1), unit(1), unit(1), unit(2), unit(1), unit(1), unit(1)}, texts: texts,
		textUnits: 5}
	var s speech
	for range 12 {
		s.add(make([]int, model.StepCodes))
	}
	for _, step := range []int{1, 3, 5, 8, 10} {
		s.started[step] = true
	}

	// From unit 4, 2 steps; from unit 3, 4; from unit 2, 7; from unit 1, 9. A longer window takes the speech from its
	// first start, not the step before it.
	for _, tc := range []struct{ window, first, from int }{{1, 4, 10}, {2, 3, 8}, {3, 3, 8}, {4, 2, 5}, {6, 2, 5},
		{7, 1, 3}, {8, 1, 3}, {9, 0, 1}, {100, 0, 1}} {
		got := filled(&s, c, tc.window)
		if !slices.Equal(got.texts, texts[tc.first:]) || len(got.units) != c.textUnits-tc.first ||
			len(got.speech.steps) != 12-tc.from || !got.speech.started[0] {
			t.Errorf("window %d: %q and %d steps, want units from %d and steps from %d", tc.window, got.texts,
				len(got.speech.steps), tc.first, tc.from)
		}
	}

	for name, tc := range map[string]struct {
		c       chunk
		started int
	}{"a chunk with no unit": {chunk{}, 1}, "more starts than units": {c, 11}} {
		s.started[tc.started] = true
		if !panics(func() { filled(&s, tc.c, 10) }) {
			t.Errorf("%s gave a tail", name)
		}
	}
}

// filled is a new tail filled from chunk c's speech s.
func filled(s *speech, c chunk, window int) *tail {
	t := new(tail)
	t.fill(s, c, window)
	return t
}

// panics reports whether f panics.
func panics(f func()) (panicked bool) {
	defer func() { panicked = recover() != nil }()
	f()
	return false
}

// TestTailPrependTo puts a tail's units before a chunk's: the units and texts of both, the tail's said first.
func TestTailPrependTo(t *testing.T) {
	unit := func(tokens int) []int { return make([]int, tokens) }
	next := chunk{units: [][]int{unit(3), unit(1), unit(1), unit(1), unit(1)}, texts: []string{"f ", "g ", "h."},
		textUnits: 3}
	spoken := (&tail{units: [][]int{unit(2), unit(1)}, texts: []string{"d ", "e."}}).prependTo(next)
	if len(spoken.units) != 2+len(next.units) || spoken.textUnits != 2+next.textUnits ||
		!slices.Equal(spoken.texts, []string{"d ", "e. ", "f ", "g ", "h."}) {
		t.Errorf("a chunk after a tail holds %d units, %d said, texts %q", len(spoken.units), spoken.textUnits,
			spoken.texts)
	}
}

// TestCarriedChunksSpeakAsOne speaks three texts as one chunk, then as three, each carrying on all of the speech
// before it, steered onto the same codes and starts. The last chunk holds every text and has read every step before
// it, so its draws and decisions must score as the one chunk's do at the same steps: a fed step that replayed the
// start that ended a chunk, a unit fed out of turn, a head that skipped the fed steps or a carry of a carried chunk
// that lost what it carried would show. A real window carries a tail whose text is only part of what was said, so
// the chunk's prompt, and with it every score, differs from one chunk's by design.
func TestCarriedChunksSpeakAsOne(t *testing.T) {
	e, g := releaseEngine(t)
	v, _ := fixtureVoice(t, e, g)
	parts, one, events := partsAndWhole(t, e, "Hello there.", "How are you?", "Fine, thanks.")
	codes := randomSteps(e, len(slices.Concat(events...)))
	ended := func(own []bool) []bool { return append(slices.Clone(own[:len(own)-1]), true) }
	wantScores, wantDecisions, _ := steer(t, e, v, one, nil, codes, ended(slices.Concat(events...)))
	var carried *tail
	at := 0
	for k, p := range parts[:len(parts)-1] {
		next := p
		if carried != nil {
			next = carried.prependTo(p)
		}

		_, _, heard := steer(t, e, v, next, carried, codes[at:at+len(events[k])], ended(events[k]))
		at += len(events[k])
		carried = filled(&heard, next, 1000)
		if len(carried.speech.steps) != at {
			t.Fatalf("chunk %d carries %v, want all %d steps", k, carried, at)
		}
	}

	last := carried.prependTo(parts[len(parts)-1])
	scores, decisions, _ := steer(t, e, v, last, carried, codes[at:], ended(events[len(events)-1]))
	for i := range scores {
		if j := firstApart(scores[i], wantScores[at*model.StepCodes+i]); j >= 0 {
			t.Fatalf("draw %d: code %d scores %f, want %f as one chunk", i, j, scores[i][j],
				wantScores[at*model.StepCodes+i][j])
		}
	}

	if firstApart(decisions, wantDecisions[at:]) >= 0 {
		t.Errorf("decisions %v, want %v as one chunk", decisions, wantDecisions[at:])
	}
}

// partsAndWhole is a chunk of each text, one chunk of them all, and each part's starts when each unit starts in the
// first of two steps.
func partsAndWhole(t *testing.T, e *Engine, texts ...string) ([]chunk, chunk, [][]bool) {
	t.Helper()
	var parts []chunk
	var events [][]bool
	var one chunk
	for _, s := range texts {
		chunks := chunksOf(t, e, s)
		if len(chunks) != 1 {
			t.Fatalf("%q: %d chunks", s, len(chunks))
		}

		p := chunks[0]
		parts = append(parts, p)
		events = append(events, slices.Repeat([]bool{true, false}, p.textUnits))
		one.units, one.texts = append(one.units, p.units[:p.textUnits]...), append(one.texts, p.texts...)
		one.textUnits += p.textUnits
	}
	one.units = append(one.units, parts[len(parts)-1].units[parts[len(parts)-1].textUnits:]...)
	return parts, one, events
}

// randomSteps is n steps of codes, each slot a code of its level.
func randomSteps(e *Engine, n int) [][]int {
	rng := rand.New(rand.NewPCG(11, 12))
	steps := make([][]int, n)
	for i := range steps {
		steps[i] = make([]int, model.StepCodes)
		for slot := range steps[i] {
			steps[i][slot] = e.settings.SlotLevels[slot%audio.FrameCodes]*audio.LevelSize + rng.IntN(audio.LevelSize)
		}
	}
	return steps
}

// steer speaks chunk after the carried tail, drawing codes and saying the starts of events, and returns the guided
// scores of every draw, every decision's logit and the speech it read.
func steer(t *testing.T, e *Engine, v *Voice, spoken chunk, carried *tail, codes [][]int,
	events []bool) ([][]float32, []float32, speech) {
	t.Helper()
	var scores [][]float32
	var decisions []float32
	c := chooser{
		draw: func(s []float32) int {
			step, slot := len(scores)/model.StepCodes, len(scores)%model.StepCodes
			scores = append(scores, slices.Clone(s))
			return codes[step][slot] % audio.LevelSize
		},
		started: func(logit float32) bool {
			decisions = append(decisions, logit)
			if len(decisions) > len(events) {
				t.Fatalf("decision %d, past the %d steered", len(decisions), len(events))
			}
			return events[len(decisions)-1]
		},
	}
	r := e.takeRequest()
	r.choose = c
	if err := r.generate(t.Context(), v, spoken, carried, func([]int) error { return nil }); err != nil {
		t.Fatal(err)
	}

	if len(scores) != len(codes)*model.StepCodes {
		t.Fatalf("%d draws, want %d", len(scores), len(codes)*model.StepCodes)
	}
	return scores, decisions, r.heard
}

// TestFeedReadsTheTrainingLayout carries the golden speech on into a second chunk. The main model must read the tail as
// training wrote a row: the opening on the tail's first units, then each step, and after a step a unit started in the
// next unit of the tail and then of the new chunk. The head must hold the tail it wrote.
func TestFeedReadsTheTrainingLayout(t *testing.T) {
	e, g := releaseEngine(t)
	v, golden := fixtureVoice(t, e, g)
	nexts := chunksOf(t, e, "Then it rained, and we went home.")
	carried := goldenSpeech(t, e, g, v, golden).tail
	spoken := carried.prependTo(nexts[0])
	r := e.takeRequest()
	r.open(v, spoken)
	if err := r.feed(t.Context(), carried); err != nil {
		t.Fatal(err)
	}

	if r.fed != len(carried.units) {
		t.Errorf("the tail fed %d units, want one for each of its %d starts", r.fed, len(carried.units))
	}

	ids, bare := trainingLayout(e, spoken, carried)
	w := e.takeRequest()
	w.openPasses(v, spoken)
	for i := range w.vectors {
		tensor.Sized(&w.vectors[i], e.model.MainHidden())
	}
	w.read(withTextPass, w.inputs(ids))
	w.read(unconditionalPass, w.inputs(bare))
	want := w.vectors
	for i := range want {
		if at := firstApart(r.vectors[i], want[i]); at >= 0 {
			t.Errorf("pass %d: value %d of the vector after the tail is %f, want %f", i, at, r.vectors[i][at],
				want[i][at])
		}
	}

	// A head that wrote none of the tail scores the next step otherwise.
	got := firstScores(e, r.headCaches[:], r.vectors[:])
	fresh := firstScores(e, []*model.Cache{e.model.NewHeadCache(), e.model.NewHeadCache()}, r.vectors[:])
	for i := range got {
		// Scores that differ count only when they are numbers: firstApart finds NaN apart from anything.
		if math.IsInf(approx.MaxDifference(got[i], fresh[i]), 1) || firstApart(got[i], fresh[i]) < 0 {
			t.Errorf("pass %d: the head scores the step after the tail as if it held none of it, or not as numbers", i)
		}
	}
}

// steeredGolden is the golden speech steered onto its codes and starts: every draw's guided scores, every decision's
// logit, and the tail the next chunk carries on.
type steeredGolden struct {
	scores    [][]float32
	decisions []float32
	tail      *tail
}

var (
	goldenOnce sync.Once
	goldenRun  *steeredGolden
)

// goldenSpeech is the golden speech steered once for every test that reads it. Only the release engine speaks it, and
// it is only read.
func goldenSpeech(t *testing.T, e *Engine, g *fixtures.Golden, v *Voice, target chunk) *steeredGolden {
	t.Helper()
	goldenOnce.Do(func() { goldenRun = steerGolden(t, e, g, v, target) })
	if goldenRun == nil {
		t.Fatal("the golden speech failed in an earlier test")
	}
	return goldenRun
}

// steerGolden steers the golden speech and keeps its tail. The speech must hold one start per unit, the one that ended
// it left out, and run past the window, so the tail is part of it.
func steerGolden(t *testing.T, e *Engine, g *fixtures.Golden, v *Voice, target chunk) *steeredGolden {
	t.Helper()
	// The golden chunk holds its units as tokens alone.
	target.texts = make([]string, target.textUnits)
	codes := slices.Collect(slices.Chunk(g.Model.Codes, model.StepCodes))
	scores, decisions, heard := steer(t, e, v, target, nil, codes, g.Model.Starts)

	if started := heard.starts(); started != target.textUnits {
		t.Fatalf("the speech holds %d starts, want one per unit, %d, without the one that ended it", started,
			target.textUnits)
	}

	carried := filled(&heard, target, e.settings.MainWindow)
	if len(carried.speech.steps) == len(heard.steps) {
		t.Fatal("the golden speech gave no tail shorter than itself")
	}
	return &steeredGolden{scores: scores, decisions: decisions, tail: carried}
}

// trainingLayout is what the two passes read of spoken after their static prompts, written as training writes a row:
// the opening, then each step of the tail, and after a step a unit started in, the next unit. The pass guidance
// compares against reads the steps alone.
func trainingLayout(e *Engine, spoken chunk, carried *tail) (ids, bare []int) {
	open, _ := spoken.split(e.settings.LookaheadUnits)
	ids, bare = e.markers.appendOpening(nil, open), e.markers.appendOpening(nil, nil)
	next := len(open)
	for at, codes := range carried.speech.steps {
		for _, code := range codes {
			ids, bare = append(ids, e.markers.AudioBase+code), append(bare, e.markers.AudioBase+code)
		}

		if carried.speech.started[at] {
			ids = append(ids, spoken.units[next]...)
			next++
		}
	}
	return ids, bare
}

// firstScores is each head's scores of the next step's first slot.
func firstScores(e *Engine, heads []*model.Cache, vectors [][]float32) [][]float32 {
	var first [][]float32
	e.model.WriteStep(new(model.State), heads, vectors, func(s [][]float32) int {
		if first == nil {
			for _, row := range s {
				first = append(first, slices.Clone(row))
			}
		}
		return 0
	})
	return first
}

// firstApart is the first index where got and want differ by more than sameTolerance, absolute and relative, or -1.
// Values of different lengths differ at the shorter one's end.
func firstApart(got, want []float32) int {
	if len(got) != len(want) {
		return min(len(got), len(want))
	}

	for i, w := range want {
		if !approx.Equal(float64(got[i]), float64(w), sameTolerance, sameTolerance) {
			return i
		}
	}
	return -1
}

// TestSpeakCarriesTheSpeechOn speaks a script of several chunks, each carrying the one before on, so the decoder must
// end the speech once, after the last chunk, and decode each join as one. The audio must be every written step decoded
// as one speech; a speech ended at the join decodes the steps around it without their context.
func TestSpeakCarriesTheSpeechOn(t *testing.T) {
	e, _ := releaseEngine(t)
	script := scriptOf(t, e, "Hello there. How are you? I am fine. Thank you.")
	silent := func(int) float32 { return 0 }
	r := e.takeRequest()
	r.seed(rand.NewPCG(1, 2))
	r.decoder = e.snac.NewStream(silent)
	// Fed steps are read with known codes, so the draws are the written steps' codes, slot after slot.
	var written []int
	draw := r.choose.draw
	r.choose.draw = func(scores []float32) int {
		code := draw(scores)
		written = append(written, e.settings.SlotLevels[len(written)%audio.FrameCodes]*audio.LevelSize+code)
		return code
	}

	var spoken []float32
	report, err := r.speakChunks(t.Context(), e.noVoice, script, func(samples []float32) error {
		spoken = append(spoken, samples...)
		return nil
	})
	if err != nil || report.Capped != 0 {
		t.Fatalf("%+v, %v", report, err)
	}

	if len(r.tail.speech.steps) == 0 {
		t.Fatal("no chunk carried the speech on")
	}

	whole := e.snac.NewStream(silent)
	var want []float32
	for at := 0; at < len(written); at += audio.FrameCodes {
		want = append(want, whole.Add(written[at:at+audio.FrameCodes])...)
	}
	want = append(want, whole.Flush()...)
	if !slices.Equal(spoken, want) {
		t.Errorf("%d samples spoken differ from the %d of every written step decoded as one speech", len(spoken),
			len(want))
	}
}

// TestNextChunkCountsTheTail packs the chunk after a carried tail: the tail's text stands first and counts against the
// sentence limit, so the chunk takes fewer sentences than it would alone.
func TestNextChunkCountsTheTail(t *testing.T) {
	e, _ := withSettings(t, func(s *settings) { s.ChunkSentences = 2 })
	script := scriptOf(t, e, "One more. Two more. Three more.")

	tailOf, err := e.encodeChunk("slower models.")
	if err != nil {
		t.Fatal(err)
	}

	carried := &tail{units: tailOf.units[:tailOf.textUnits], texts: tailOf.texts}
	for _, tc := range []struct {
		carried *tail
		want    string
		left    int
	}{{nil, "One more. Two more.", 1}, {carried, "slower models. One more.", 2}} {
		chunk, left, err := e.nextChunk(script.pieces, tc.carried)
		if err != nil {
			t.Fatal(err)
		}

		if got := strings.Join(chunk.texts, ""); got != tc.want || len(left) != tc.left {
			t.Errorf("after %v: %q with %d pieces left, want %q with %d", tc.carried != nil, got, len(left),
				tc.want, tc.left)
		}
	}
}
