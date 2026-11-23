package model

import (
	"fmt"
	"math"
	"slices"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// transformer is the Qwen3.5 text model with every layer full attention, as each part of the speech model is built.
// It reads input vectors, not tokens: the speech model makes them from tokens or from a step's codes. It only reads
// its weights; a pass writes to the partState its caller gives it.
type transformer struct {
	hidden             int
	inverseFrequencies []float32
	norm               rmsNorm
	layers             []layer
}

// partState is the memory a part's forward passes write, each grown to the largest pass so far and kept from pass to
// pass: the head makes a pass for each code a step writes, and fresh buffers cost about a tenth of a step to allocate,
// zero and free. The zero value is ready to use.
type partState struct {
	x, normed, states, cos, sin, qkv, attended, attentionOut, gateUp, activated, feedForwardOut []float32
	// Each input position's cache and how many positions of the cache and input it attends to, the same in every layer.
	positionCache   []*Cache
	positionVisible []int
	widest          int
	// Each worker's attention scores and gate values, and the task that carries a layer's attention to the workers.
	scores, gates [][]float32
	attentionTask attentionTask
}

// layer is one transformer block: attention, then the feed-forward, each reading a normed copy of the token stream
// and returning what to add back to it.
type layer struct {
	attentionNorm, feedForwardNorm rmsNorm
	attention                      attention
	feedForward                    feedForward
}

// attention keeps q with each head's output gate, k and v as one matrix, so each token's projections are one pass over
// the weights.
type attention struct {
	cacheSlot                       int
	hidden, heads, kvHeads, headDim int
	scale                           float32
	qkv, o                          tensor.Matrix
	qNorm, kNorm                    rmsNorm
}

// feedForward keeps gate and up as one matrix, so each token's projections are one pass over the weights.
type feedForward struct {
	hidden, size int
	gateUp, down tensor.Matrix
}

// transformerConfig is the part of a Hugging Face Qwen3.5 text config the engine reads.
type transformerConfig struct {
	HiddenSize       int        `json:"hidden_size"`
	IntermediateSize int        `json:"intermediate_size"`
	Layers           int        `json:"num_hidden_layers"`
	Heads            int        `json:"num_attention_heads"`
	KVHeads          int        `json:"num_key_value_heads"`
	HeadDim          int        `json:"head_dim"`
	Epsilon          float64    `json:"rms_norm_eps"`
	Vocab            int        `json:"vocab_size"`
	HiddenAct        string     `json:"hidden_act"`
	AttentionBias    bool       `json:"attention_bias"`
	LayerTypes       []string   `json:"layer_types"`
	Rope             ropeConfig `json:"rope_parameters"`
}

type ropeConfig struct {
	Type    string  `json:"rope_type"`
	Theta   float64 `json:"rope_theta"`
	Partial float64 `json:"partial_rotary_factor"`
}

// check refuses a config the engine cannot run, naming the first setting at fault.
func (c transformerConfig) check() error {
	notFullAttention := func(t string) bool { return t != "full_attention" }
	for _, rule := range []struct {
		setting string
		refused bool
	}{
		{"hidden_size", c.HiddenSize <= 0},
		{"intermediate_size", c.IntermediateSize <= 0},
		{"num_hidden_layers", c.Layers <= 0},
		{"num_attention_heads", c.Heads <= 0},
		{"num_key_value_heads", c.KVHeads <= 0 || c.Heads%max(c.KVHeads, 1) != 0},
		{"head_dim", c.HeadDim <= 0},
		{"vocab_size", c.Vocab <= 0},
		{"rms_norm_eps", c.Epsilon <= 0},
		{"hidden_act", c.HiddenAct != "silu"},
		{"attention_bias", c.AttentionBias},
		{"rope_parameters", c.Rope.Type != "default" || c.Rope.Theta <= 0 || c.Rope.Partial <= 0 || c.Rope.Partial > 1},
		{"layer_types", len(c.LayerTypes) != c.Layers || slices.ContainsFunc(c.LayerTypes, notFullAttention)},
	} {
		if rule.refused {
			return fmt.Errorf("%s is out of range or unsupported", rule.setting)
		}
	}
	return nil
}

// loadTransformer reads the layers named prefix+"layers.N." and the final norm prefix+"norm.weight".
func loadTransformer(l *tensor.Loader, prefix string, config transformerConfig) *transformer {
	h, d := config.HiddenSize, config.HeadDim
	epsilon := config.Epsilon
	m := &transformer{hidden: h, norm: loadRMSNorm(l, prefix+"norm.weight", h, epsilon)}
	rotary := int(float64(d) * config.Rope.Partial)
	for i := 0; i < rotary; i += 2 {
		frequency := 1 / math.Pow(config.Rope.Theta, float64(i)/float64(rotary))
		m.inverseFrequencies = append(m.inverseFrequencies, float32(frequency))
	}

	for i := range config.Layers {
		layerPrefix := fmt.Sprintf("%slayers.%d.", prefix, i)
		matrixPart := func(name string, rows int) tensor.MatrixPart {
			return tensor.MatrixPart{Name: layerPrefix + name, Rows: rows}
		}
		m.layers = append(m.layers, layer{
			attentionNorm: loadRMSNorm(l, layerPrefix+"input_layernorm.weight", h, epsilon),
			attention: attention{cacheSlot: i, hidden: h, heads: config.Heads, kvHeads: config.KVHeads, headDim: d,
				scale: float32(1 / math.Sqrt(float64(d))),
				qkv: l.Matrices(h, matrixPart("self_attn.q_proj.weight", 2*config.Heads*d),
					matrixPart("self_attn.k_proj.weight", config.KVHeads*d),
					matrixPart("self_attn.v_proj.weight", config.KVHeads*d)),
				o:     l.Matrix(layerPrefix+"self_attn.o_proj.weight", h, config.Heads*d),
				qNorm: loadRMSNorm(l, layerPrefix+"self_attn.q_norm.weight", d, epsilon),
				kNorm: loadRMSNorm(l, layerPrefix+"self_attn.k_norm.weight", d, epsilon)},
			feedForwardNorm: loadRMSNorm(l, layerPrefix+"post_attention_layernorm.weight", h, epsilon),
			feedForward: feedForward{hidden: h, size: config.IntermediateSize,
				gateUp: l.Matrices(h, matrixPart("mlp.gate_proj.weight", config.IntermediateSize),
					matrixPart("mlp.up_proj.weight", config.IntermediateSize)),
				down: l.Matrix(layerPrefix+"mlp.down_proj.weight", h, config.IntermediateSize)},
		})
	}
	return m
}

// rmsNorm scales each row of its input to unit root mean square, then by its gains.
type rmsNorm struct {
	gains   []float32
	epsilon float64
}

// loadEntryNorm reads the norm that vectors entering part's stream go through, with the part's epsilon.
func loadEntryNorm(l *tensor.Loader, name string, part *transformer) rmsNorm {
	return loadRMSNorm(l, name+".weight", part.hidden, part.norm.epsilon)
}

// loadRMSNorm reads the weights of a norm over rows of size values. They are stored around zero, so a weight of 0
// keeps the input's scale.
func loadRMSNorm(l *tensor.Loader, name string, size int, epsilon float64) rmsNorm {
	weights := l.Float32s(name, size)
	gains := make([]float32, len(weights))
	for i, w := range weights {
		gains[i] = 1 + w
	}
	return rmsNorm{gains: gains, epsilon: epsilon}
}

// forward runs each of inputs after the positions its cache holds and returns the final normed state of every
// position, the inputs' positions one after another. A position attends to its cache and, when causal, to its own
// input up to itself; otherwise to all of its input. The inputs share each read of a weight, as a batch. The states
// are state's memory, read until its next pass.
func (part *transformer) forward(state *partState, caches []*Cache, inputs [][]float32, causal bool) []float32 {
	if len(caches) != len(inputs) {
		panic(fmt.Sprintf("%d caches for %d inputs", len(caches), len(inputs)))
	}

	for i, cache := range caches {
		if slices.Contains(caches[:i], cache) {
			panic("two inputs share a cache, so each would attend to the other's positions")
		}
	}

	h := part.hidden
	x := joined(&state.x, inputs)
	n := len(x)

	part.positions(state, caches, inputs, causal)
	cos, sin := part.angles(state, caches, inputs)
	normed := tensor.Sized(&state.normed, n)
	for i := range part.layers {
		current := &part.layers[i]
		current.attentionNorm.apply(normed, x)
		tensor.Add(x, current.attention.apply(state, normed, cos, sin))
		current.feedForwardNorm.apply(normed, x)
		tensor.Add(x, current.feedForward.apply(state, normed))
	}

	for i, cache := range caches {
		cache.advance(len(inputs[i]) / h)
	}

	states := tensor.Sized(&state.states, n)
	part.norm.apply(states, x)
	// The pass's caches are its caller's, so the partState holds on to none of them.
	clear(state.positionCache)
	return states
}

// positions sets each input position's cache and how many positions of the cache and input it attends to, and the most
// any position attends to.
func (part *transformer) positions(state *partState, caches []*Cache, inputs [][]float32, causal bool) {
	state.positionCache, state.positionVisible = state.positionCache[:0], state.positionVisible[:0]
	state.widest = 0
	for i, cache := range caches {
		length := len(inputs[i]) / part.hidden
		state.widest = max(state.widest, cache.length+length)
		for t := range length {
			visible := cache.length + length
			if causal {
				visible = cache.length + t + 1
			}
			state.positionCache = append(state.positionCache, cache)
			state.positionVisible = append(state.positionVisible, visible)
		}
	}
}

// angles is the cos and sin of each rotary frequency at each input position, the rows one after another, which every
// layer and head shares.
func (part *transformer) angles(state *partState, caches []*Cache, inputs [][]float32) (cos, sin []float32) {
	cos, sin = state.cos[:0], state.sin[:0]
	for i, cache := range caches {
		for t := range len(inputs[i]) / part.hidden {
			for _, f := range part.inverseFrequencies {
				// The product is f32, as training rotated by it; only the sine and cosine are f64.
				angle := float64(f * float32(cache.position+t))
				cos, sin = append(cos, float32(math.Cos(angle))), append(sin, float32(math.Sin(angle)))
			}
		}
	}
	state.cos, state.sin = cos, sin
	return cos, sin
}

// apply runs attention for the positions of x, each after the cache state.positionCache gives it, cos and sin holding
// each position's rotary angles.
func (a *attention) apply(state *partState, x, cos, sin []float32) []float32 {
	h, d, heads, kvHeads := a.hidden, a.headDim, a.heads, a.kvHeads
	n := len(x) / h
	// Each token's row holds every head's query followed by its output gate, then the keys, then the values.
	width := (2*heads + 2*kvHeads) * d
	qkv := tensor.LinearInto(&state.qkv, x, a.qkv)

	half := len(cos) / n
	for r, cache := range state.positionCache {
		qkvRow := qkv[r*width:][:width]
		rowCos, rowSin := cos[r*half:][:half], sin[r*half:][:half]
		for head := range heads {
			q := qkvRow[head*2*d:][:d]
			a.qNorm.apply(q, q)
			rope(q, rowCos, rowSin)
		}

		keys := qkvRow[2*heads*d:][:kvHeads*d]
		for head := range kvHeads {
			k := keys[head*d:][:d]
			a.kNorm.apply(k, k)
			rope(k, rowCos, rowSin)
		}
		cache.appendLayer(a.cacheSlot, keys, qkvRow[(2*heads+kvHeads)*d:])
	}

	// One position and head per work item. attend adds into its output.
	out := tensor.Sized(&state.attended, n*heads*d)
	clear(out)
	tensor.SizedPerWorker(&state.scores, state.widest)
	tensor.SizedPerWorker(&state.gates, d)
	task := attentionTask{a: a, state: state, qkv: qkv, out: out, width: width}
	tensor.RunParallel(&state.attentionTask, task, n*heads, state.widest*d)

	return tensor.LinearInto(&state.attentionOut, out, a.o)
}

// attentionTask carries a layer's attention to the workers, each scoring in its own memory.
type attentionTask struct {
	a        *attention
	state    *partState
	qkv, out []float32
	width    int
}

func (task *attentionTask) Run(lo, hi, worker int) {
	a, state, d, width := task.a, task.state, task.a.headDim, task.width
	scores, gates := state.scores[worker][:state.widest], state.gates[worker][:d]
	for item := lo; item < hi; item++ {
		r, head := item/a.heads, item%a.heads
		q, gate := task.qkv[r*width+head*2*d:][:d], task.qkv[r*width+head*2*d+d:][:d]
		o := task.out[item*d:][:d]
		cache := state.positionCache[r]
		total := a.attend(o, q, cache.k[a.cacheSlot], cache.v[a.cacheSlot], head, scores[:state.positionVisible[r]])
		tensor.SigmoidRow(gates, gate)
		for j, g := range gates {
			o[j] *= g / total
		}
	}
}

// attend adds into out the values of the first len(scores) cached tokens, each weighted by its unnormalized softmax
// weight for q, and returns the sum of the weights; the caller divides by it.
func (a *attention) attend(out, q []float32, keys, values []uint16, head int, scores []float32) float32 {
	d, kvHeads := a.headDim, a.kvHeads
	kvHead := head / (a.heads / kvHeads)
	stride := kvHeads * d
	key := func(p int) []uint16 { return keys[p*stride+kvHead*d:][:d] }
	p := 0
	for ; p+4 <= len(scores); p += 4 {
		scores[p], scores[p+1], scores[p+2], scores[p+3] = tensor.Dot4BF16(q, key(p), key(p+1), key(p+2), key(p+3))
	}

	for ; p < len(scores); p++ {
		scores[p] = tensor.DotBF16(q, key(p))
	}

	tensor.ExpFromMax(scores, scores, a.scale)
	total := tensor.Sum(scores)

	tensor.AccumulateRowsBF16(out, scores, values[kvHead*d:], stride)
	return total
}

// rope rotates the first rotary dimensions of a head as two halves, leaving the rest unrotated.
func rope(x, cos, sin []float32) {
	half := len(cos)
	for i := range cos {
		first, second := x[i], x[i+half]
		x[i] = first*cos[i] - second*sin[i]
		x[i+half] = second*cos[i] + first*sin[i]
	}
}

func (f *feedForward) apply(state *partState, x []float32) []float32 {
	size, n := f.size, len(x)/f.hidden
	gateUp := tensor.LinearInto(&state.gateUp, x, f.gateUp)
	activated := tensor.Sized(&state.activated, n*size)
	for t := range n {
		gate, up, out := gateUp[t*2*size:][:size], gateUp[t*2*size+size:][:size], activated[t*size:][:size]
		tensor.SiLURow(out, gate)
		for i, u := range up {
			out[i] *= u
		}
	}

	return tensor.LinearInto(&state.feedForwardOut, activated, f.down)
}

// apply norms each row of x into out, which may be x.
func (n rmsNorm) apply(out, x []float32) {
	size := len(n.gains)
	for row := range len(x) / size {
		values, outRow := x[row*size:][:size], out[row*size:][:size]
		scale := float32(1 / math.Sqrt(float64(tensor.Dot(values, values))/float64(size)+n.epsilon))
		for i, v := range values {
			outRow[i] = v * scale * n.gains[i]
		}
	}
}
