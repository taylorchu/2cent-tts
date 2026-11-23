// Package model is the speech model, which reads text and writes SNAC codes a step at a time, and the Qwen3.5
// transformer each of its parts is built from. It takes its kernels and weights from the tensor package and the
// codec's frame layout from the audio package; it knows nothing of text or of generation.
package model

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/taylorchu/2cent-tts/internal/audio"
	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// The model writes and reads speech a step at a time: StepFrames SNAC frames, which hold StepCodes codes.
const (
	StepFrames = 2
	StepCodes  = StepFrames * audio.FrameCodes
)

// Model writes speech a step of SNAC frames at a time, in four parts:
//   - the main model reads one position for each text token or step and gives a vector for each;
//   - the head draws a step's codes from the main model's vector, one code after another;
//   - the reader turns a step's codes into the main model's input for that step;
//   - the decision head says, of a step just written, whether a unit of the text started being spoken in it.
type Model struct {
	main, head, reader *transformer
	// tokenTable holds a row of the main model's input for each token except the codes, which only the reader reads.
	tokenTable tensor.Matrix
	// codeTable holds a row for each code: the reader's and the head's input, and the head's output weights.
	codeTable                           tensor.Matrix
	toMain, toHead, decideIn, decideOut tensor.Matrix
	// Every vector enters its part's token stream through a norm of its own, as the model was trained: the main
	// model's tokens and steps, the reader's codes, and the head's vector and codes. The head's output weights read
	// the code table raw.
	mainToken, mainStep, readerCode, headVector, headCode rmsNorm
	layout                                                Layout
}

// State is the memory the model's passes and steps write, kept from step to step so they allocate nothing. A Model
// only reads its weights, so any number of goroutines may share one, each passing a State of its own; a State serves
// one goroutine at a time. A slice a method returns in it lives until that method, or one that calls it, runs again
// with the same State: StepInput's until the next StepInput, Inputs or AppendInputs, WriteStep's codes until the next
// WriteStep. The zero value is ready to use.
type State struct {
	main, head, reader     partState
	joined                 []float32
	codes                  []int
	logits                 []float32
	scores                 [][]float32
	inputs                 [][]float32
	passInputs             [][]float32
	headInput              []float32
	codeInput              []float32
	readerInput, mainInput []float32
	readerCache            *Cache
	decideHidden           []float32
	decideActivated        []float32
	step                   [StepCodes]int
	decision               [1]float32
}

// Layout is how the model lays codes out, as it was trained.
type Layout struct {
	// AudioBase is the token id of the first code.
	AudioBase int
	// SlotLevels is the SNAC level of each slot of a frame, in the order the head writes them.
	SlotLevels []int
	// PastSteps is how many steps before the one it writes the head attends to.
	PastSteps int
}

// Load reads the model of the transformer configs in configData, laid out as layout says. A missing or misshapen tensor
// is the loader's error, which Load returns too.
func Load(l *tensor.Loader, configData []byte, layout Layout) (*Model, error) {
	err := audio.CheckSlotLevels(layout.SlotLevels)
	if err != nil {
		return nil, err
	}

	if layout.PastSteps < 0 {
		return nil, fmt.Errorf("the head attends to %d past steps", layout.PastSteps)
	}

	var configs map[string]transformerConfig
	err = json.Unmarshal(configData, &configs)
	if err != nil {
		return nil, err
	}

	for _, name := range []string{"main", "head", "reader"} {
		config, ok := configs[name]
		if !ok {
			return nil, fmt.Errorf("no %s config", name)
		}

		err = config.check()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
	}

	if configs["head"].Vocab != audio.CodeCount {
		return nil, fmt.Errorf("the head holds %d codes, want %d", configs["head"].Vocab, audio.CodeCount)
	}

	// The reader reads the codes from the head's own table.
	if configs["reader"].HiddenSize != configs["head"].HiddenSize {
		return nil, fmt.Errorf("the reader is %d wide and the head %d; they share one code table",
			configs["reader"].HiddenSize, configs["head"].HiddenSize)
	}

	main := loadTransformer(l, "main.", configs["main"])
	head := loadTransformer(l, "head.", configs["head"])
	reader := loadTransformer(l, "reader.", configs["reader"])
	decideSize := l.Dims("decide_in.weight", 2)[0]
	m := &Model{main: main, head: head, reader: reader, layout: layout,
		tokenTable: l.Matrix("main.embed_tokens.weight", configs["main"].Vocab, main.hidden),
		codeTable:  l.Matrix("head.embed_tokens.weight", audio.CodeCount, head.hidden),
		toMain:     l.Matrix("to_main.weight", main.hidden, StepCodes*reader.hidden),
		toHead:     l.Matrix("to_head.weight", head.hidden, main.hidden),
		decideIn:   l.Matrix("decide_in.weight", decideSize, main.hidden),
		decideOut:  l.Matrix("decide_out.weight", 1, decideSize),
		mainToken:  loadEntryNorm(l, "main_token_norm", main),
		mainStep:   loadEntryNorm(l, "main_step_norm", main),
		readerCode: loadEntryNorm(l, "reader_code_norm", reader),
		headVector: loadEntryNorm(l, "head_vector_norm", head),
		headCode:   loadEntryNorm(l, "head_code_norm", head)}
	if l.Err() != nil {
		return nil, l.Err()
	}
	return m, nil
}

// MainHidden is the length of the main model's vectors.
func (m *Model) MainHidden() int {
	return m.main.hidden
}

// NewMainCache is an empty cache for the main model.
func (m *Model) NewMainCache() *Cache {
	return m.main.newCache()
}

// NewHeadCache is an empty cache for the head.
func (m *Model) NewHeadCache() *Cache {
	return m.head.newCache()
}

// ReadMain runs inputs through the main model after the positions cache holds, each attending to those and to the
// inputs up to its own, and copies the last one's vector into vector, which may be nil when no one needs it.
func (m *Model) ReadMain(st *State, cache *Cache, vector []float32, inputs [][]float32) {
	states := m.main.forward(&st.main, []*Cache{cache}, [][]float32{joined(&st.joined, inputs)}, true)
	copy(vector, states[len(states)-m.main.hidden:])
}

// ReadMainStep runs one step's input through the main model once after the positions of each of caches, and copies
// each pass's vector into vectors[i]. The passes share each read of the weights. It panics if the input is not one
// position or two of caches are the same cache.
func (m *Model) ReadMainStep(st *State, caches []*Cache, vectors [][]float32, input []float32) {
	h := m.main.hidden
	st.passInputs = st.passInputs[:0]
	for range caches {
		st.passInputs = append(st.passInputs, input)
	}
	onePosition(st.passInputs, h)
	states := m.main.forward(&st.main, caches, st.passInputs, true)
	for i, vector := range vectors {
		copy(vector, states[i*h:][:h])
	}
}

// joined is parts one after another in buf's memory.
func joined(buf *[]float32, parts [][]float32) []float32 {
	*buf = (*buf)[:0]
	for _, part := range parts {
		*buf = append(*buf, part...)
	}
	return *buf
}

// onePosition panics unless each input holds one position: a longer one would return its first position's vector, not
// its last.
func onePosition(inputs [][]float32, hidden int) {
	for _, input := range inputs {
		if len(input) != hidden {
			panic(fmt.Sprintf("an input holds %d values, want one position of %d", len(input), hidden))
		}
	}
}

// WriteStep writes a step's codes from the main model's vectors, a batch of them, with headCaches[i] holding the last
// steps of vectors[i]. pick is called once for each slot, in slot order, with every head's scores of the slot's SNAC
// level, and every head reads the chosen code before the next slot is scored. The batch shares each read of the
// weights.
//
// The heads then keep only their last PastSteps steps, as the trained head saw no more. The main model's window is
// cut by the engine instead, since only the engine knows where a prompt's static part ends.
func (m *Model) WriteStep(st *State, headCaches []*Cache, vectors [][]float32,
	pick func(scores [][]float32) int) []int {
	h := m.head.hidden
	// The vectors go through toHead in one call, sharing each read of its weights.
	headInput := tensor.LinearInto(&st.headInput, joined(&st.joined, vectors), m.toHead)
	m.headVector.apply(headInput, headInput)
	tensor.Sized(&st.inputs, len(vectors))
	for i := range st.inputs {
		st.inputs[i] = headInput[i*h:][:h]
	}

	codes := st.codes[:0]
	logits := tensor.Sized(&st.logits, len(vectors)*audio.LevelSize)
	tensor.Sized(&st.scores, len(vectors))
	for i := range st.scores {
		st.scores[i] = logits[i*audio.LevelSize:][:audio.LevelSize]
	}

	for slot := range StepCodes {
		if slot > 0 {
			// forward copies its inputs, so every head can read the one input of the code before.
			code := tensor.Sized(&st.codeInput, h)
			m.codeTable.Row(code, codes[slot-1])
			m.headCode.apply(code, code)
			for i := range st.inputs {
				st.inputs[i] = code
			}
		}

		level := m.layout.SlotLevels[slot%audio.FrameCodes]
		m.headScores(st, headCaches, st.inputs, level, logits)
		codes = append(codes, level*audio.LevelSize+pick(st.scores))
	}
	st.codes = codes

	for _, cache := range headCaches {
		cache.KeepWindow(0, m.layout.PastSteps*StepCodes)
	}
	return codes
}

// headScores runs inputs[i], one position, through the head after the positions caches[i] holds and sets logits to
// each one's scores of a level's codes, one input's after another. The inputs share each read of the weights.
func (m *Model) headScores(st *State, caches []*Cache, inputs [][]float32, level int, logits []float32) {
	onePosition(inputs, m.head.hidden)
	states := m.head.forward(&st.head, caches, inputs, true)
	tensor.Linear(logits, states, m.codeTable.Rows(level*audio.LevelSize, (level+1)*audio.LevelSize))
}

// Inputs is the main model's input for each position of ids: a token, or a step of codes, which the reader reads.
// Codes come only from references, which are cut to whole steps, so codes that end partway through a step panic.
func (m *Model) Inputs(st *State, ids []int) [][]float32 {
	inputs, _ := m.AppendInputs(st, nil, nil, ids)
	return inputs
}

// AppendInputs is Inputs appended to inputs, each input's values appended to values, so a caller that keeps both
// from call to call makes its inputs without allocating. An input already appended stays valid when values grows,
// since growing copies the values elsewhere and leaves the old ones as they were.
func (m *Model) AppendInputs(st *State, inputs [][]float32, values []float32, ids []int) ([][]float32, []float32) {
	h := m.main.hidden
	for at := 0; at < len(ids); {
		start := len(values)
		values = slices.Grow(values, h)[:start+h]
		input := values[start : start+h : start+h]
		if code := ids[at] - m.layout.AudioBase; code < 0 || code >= audio.CodeCount {
			m.tokenInput(input, ids[at])
			inputs = append(inputs, input)
			at++
			continue
		}

		if at+StepCodes > len(ids) {
			panic(fmt.Sprintf("%d codes end partway through a step", len(ids)-at))
		}

		step := st.step[:]
		for i := range step {
			step[i] = ids[at+i] - m.layout.AudioBase
		}
		copy(input, m.StepInput(st, step))
		inputs = append(inputs, input)
		at += StepCodes
	}
	return inputs, values
}

// tokenInput writes a token's row into out. Ids above the codes move down by their count, since the table holds no
// row for one.
func (m *Model) tokenInput(out []float32, id int) {
	if id >= m.layout.AudioBase {
		id -= audio.CodeCount
	}

	m.tokenTable.Row(out, id)
	m.mainToken.apply(out, out)
}

// StepInput is a step's codes, each a row of the code table, read by the reader with every code attending to every
// other, and its states side by side through one matrix.
func (m *Model) StepInput(st *State, step []int) []float32 {
	h := m.reader.hidden
	x := tensor.Sized(&st.readerInput, StepCodes*h)
	for i, code := range step {
		m.codeTable.Row(x[i*h:(i+1)*h], code)
	}
	m.readerCode.apply(x, x)

	// The reader attends within the step alone, so its cache starts empty each time.
	if st.readerCache == nil {
		st.readerCache = m.reader.newCache()
	}
	st.readerCache.Reset()
	states := m.reader.forward(&st.reader, []*Cache{st.readerCache}, [][]float32{x}, false)
	out := tensor.LinearInto(&st.mainInput, states, m.toMain)
	m.mainStep.apply(out, out)
	return out
}

// Decide is the decision's logit for the vector of a step just written: above zero, a unit started in it.
func (m *Model) Decide(st *State, vector []float32) float32 {
	hidden := tensor.LinearInto(&st.decideHidden, vector, m.decideIn)
	activated := tensor.Sized(&st.decideActivated, len(hidden))
	tensor.SiLURow(activated, hidden)

	tensor.Linear(st.decision[:], activated, m.decideOut)
	return st.decision[0]
}

// TokenCount is how many token ids the main model reads: a row of its table for each token except the codes, and the
// codes, which the reader reads.
func (m *Model) TokenCount() int {
	return m.tokenTable.RowCount() + audio.CodeCount
}
