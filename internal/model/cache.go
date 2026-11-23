package model

import (
	"slices"

	"github.com/taylorchu/2cent-tts/internal/tensor"
)

// Cache holds each layer's keys and values in bf16, as attention computed them in training, at half the bytes every
// generated vector reads. A Cache serves one goroutine at a time.
type Cache struct {
	k, v [][]uint16
	// width is how many key values one position stores in one layer; its values take as many.
	width int
	// The cache can drop positions from its middle, so position, the number the next one is rotated by, is kept apart
	// from length, how many it holds.
	length, position int
}

func (part *transformer) newCache() *Cache {
	a := part.layers[0].attention
	return &Cache{k: make([][]uint16, len(part.layers)), v: make([][]uint16, len(part.layers)),
		width: a.kvHeads * a.headDim}
}

// appendLayer appends one layer's keys and values for positions after the cached ones.
func (c *Cache) appendLayer(layer int, keys, values []float32) {
	c.k[layer] = tensor.AppendBF16(c.k[layer], keys)
	c.v[layer] = tensor.AppendBF16(c.v[layer], values)
}

// advance counts positions that every layer has appended.
func (c *Cache) advance(positions int) {
	c.length += positions
	c.position += positions
}

// Reset empties the cache, keeping its memory.
func (c *Cache) Reset() {
	for i := range c.k {
		c.k[i], c.v[i] = c.k[i][:0], c.v[i][:0]
	}
	c.length, c.position = 0, 0
}

// Len is how many positions the cache holds.
func (c *Cache) Len() int {
	return c.length
}

// CopyFrom makes c a copy of src in c's own memory, with room for room more positions before it grows, so a cache
// reused chunk after chunk allocates only when a chunk needs more than any before it.
func (c *Cache) CopyFrom(src *Cache, room int) {
	if len(c.k) != len(src.k) {
		c.k, c.v = make([][]uint16, len(src.k)), make([][]uint16, len(src.v))
	}

	for i := range src.k {
		c.k[i] = copyWithRoom(c.k[i], src.k[i], room*src.width)
		c.v[i] = copyWithRoom(c.v[i], src.v[i], room*src.width)
	}
	c.width, c.length, c.position = src.width, src.length, src.position
}

func copyWithRoom(dst, src []uint16, room int) []uint16 {
	if cap(dst) < len(src)+room {
		dst = make([]uint16, 0, len(src)+room)
	}
	return append(dst[:0], src...)
}

// KeepWindow drops every cached position except the first static ones and the newest window, which is all a position
// attends to in training.
func (c *Cache) KeepWindow(static, window int) {
	if c.length <= static+window {
		return
	}

	for i := range c.k {
		c.k[i] = slices.Delete(c.k[i], static*c.width, (c.length-window)*c.width)
		c.v[i] = slices.Delete(c.v[i], static*c.width, (c.length-window)*c.width)
	}
	c.length = static + window
}
