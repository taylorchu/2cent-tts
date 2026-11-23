package model

import (
	"slices"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// TestKeepWindow keeps a cache's static positions and its newest window, its rotary position unchanged, and leaves a
// cache within its window as it was.
func TestKeepWindow(t *testing.T) {
	cache := &Cache{k: [][]uint16{nil}, v: [][]uint16{nil}, width: 2}
	for p := range 10 {
		cache.k[0] = append(cache.k[0], uint16(p), uint16(p))
		cache.v[0] = append(cache.v[0], uint16(p), uint16(p))
	}
	cache.length, cache.position = 10, 10
	cache.KeepWindow(3, 4)
	if want := []uint16{0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 8, 8, 9, 9}; !slices.Equal(cache.k[0], want) ||
		!slices.Equal(cache.v[0], want) || cache.length != 7 || cache.position != 10 {
		t.Errorf("kept %v, length %d, position %d", cache.k[0], cache.length, cache.position)
	}

	cache.KeepWindow(3, 4)
	if cache.length != 7 {
		t.Errorf("a cache within its window lost positions: length %d", cache.length)
	}
}

// TestLoadMissingTensors loads from a weights file with no tensors: every loader must report it, not panic.
func TestLoadMissingTensors(t *testing.T) {
	part := `{"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 1,
		"num_key_value_heads": 1, "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": 12288, "hidden_act": "silu",
		"layer_types": ["full_attention"],
		"rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5}}`
	config := `{"main": ` + part + `, "head": ` + part + `, "reader": ` + part + `}`
	l := tensor.NewLoader(&tensor.Weights{})
	m, err := Load(l, []byte(config), Layout{AudioBase: 5, SlotLevels: []int{0, 1, 2, 2, 1, 2, 2}})
	if err == nil || m != nil {
		t.Error("loading from no tensors returned a model and no error")
	}
}

// TestInputsRefusesAPartialStep panics on codes that end partway through a step.
func TestInputsRefusesAPartialStep(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("codes ending partway through a step were read")
		}
	}()
	(&Model{layout: Layout{AudioBase: 10}}).Inputs(new(State), []int{10, 11, 12})
}

// TestConfigCheck accepts a config the model supports and refuses each setting out of range or unsupported.
func TestConfigCheck(t *testing.T) {
	good := transformerConfig{HiddenSize: 8, IntermediateSize: 16, Layers: 2, Heads: 2, KVHeads: 1, HeadDim: 8,
		Epsilon: 1e-6, Vocab: 16, HiddenAct: "silu", LayerTypes: []string{"full_attention", "full_attention"},
		Rope: ropeConfig{Type: "default", Theta: 10000, Partial: 1}}
	if err := good.check(); err != nil {
		t.Fatal(err)
	}

	for name, change := range map[string]func(*transformerConfig){
		"fewer layer types than layers": func(c *transformerConfig) { c.LayerTypes = c.LayerTypes[:1] },
		"no hidden size":                func(c *transformerConfig) { c.HiddenSize = 0 },
		"no head size":                  func(c *transformerConfig) { c.HeadDim = 0 },
		"another activation":            func(c *transformerConfig) { c.HiddenAct = "gelu" },
		"heads sharing keys unevenly":   func(c *transformerConfig) { c.Heads, c.KVHeads = 3, 2 },
		"no key heads":                  func(c *transformerConfig) { c.KVHeads = 0 },
		"no rotary base":                func(c *transformerConfig) { c.Rope.Theta = 0 },
		"no norm epsilon":               func(c *transformerConfig) { c.Epsilon = 0 },
		"a negative vocabulary":         func(c *transformerConfig) { c.Vocab = -5 },
		"another layer type":            func(c *transformerConfig) { c.LayerTypes[1] = "linear_attention" },
	} {
		config := good
		config.LayerTypes = slices.Clone(good.LayerTypes)
		change(&config)
		if config.check() == nil {
			t.Errorf("a config with %s was accepted", name)
		}
	}
}
