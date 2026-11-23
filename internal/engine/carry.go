package engine

import (
	"fmt"
	"slices"
	"strings"
)

// speech is the steps a chunk's main model read, fed or written: each step's codes and whether a unit started in it.
// The start that ended the speech is not among them, since a chunk that carries the speech on reads that step with no
// start.
type speech struct {
	steps [][]int
	// codes holds the steps' codes back to back. A step still points into the old array after codes grows, which is
	// safe because nothing writes a step.
	codes   []int
	started []bool
}

// add appends a copy of a step in which no unit started yet.
func (s *speech) add(step []int) {
	at := len(s.codes)
	s.codes = append(s.codes, step...)
	s.steps = append(s.steps, s.codes[at:len(s.codes):len(s.codes)])
	s.started = append(s.started, false)
}

// starts is how many units started in s.
func (s *speech) starts() int {
	n := 0
	for _, started := range s.started {
		if started {
			n++
		}
	}
	return n
}

// empty takes every step out of s, keeping its memory for the next speech.
func (s *speech) empty() {
	s.steps, s.codes, s.started = s.steps[:0], s.codes[:0], s.started[:0]
}

// tail is the end of a chunk's speech that the next chunk feeds before it draws a code: its last units, as tokens and
// as text, and the steps spoken from the one the first of them started in. A request keeps one, refilled for each
// chunk.
type tail struct {
	units  [][]int
	texts  []string
	speech speech
}

// fill copies the end of chunk c's speech s into t, for the next chunk to feed. Speech older than the window is never
// seen once the next chunk draws, so the tail starts at the latest unit start after which more than window steps were
// spoken, or at the first start when the speech is shorter.
//
// Counting steps alone is what the unconditional pass needs, since it reads the steps alone: that many fill its window
// as uncut speech would. The pass with text, which reads the fed units too, is full sooner.
//
// generate decides one start for each unit of c before the start that ends the speech, and pack refuses a chunk with no
// unit, so the starts always account for c's units.
func (t *tail) fill(s *speech, c chunk, window int) {
	starts := s.starts()
	if c.textUnits == 0 || starts != c.textUnits {
		panic(fmt.Sprintf("%d starts for a chunk of %d units", starts, c.textUnits))
	}

	// From the last start back, each start's unit and step, until one has more than window steps after it.
	first, from := starts, 0
	for step, started := range slices.Backward(s.started) {
		if !started {
			continue
		}

		first, from = first-1, step
		if len(s.steps)-step > window {
			break
		}
	}

	t.units, t.texts = c.units[first:c.textUnits], c.texts[first:c.textUnits]
	t.speech.empty()
	for at, step := range s.steps[from:] {
		t.speech.add(step)
		t.speech.started[at] = s.started[from+at]
	}
}

// text is what the tail says, which the chunk after it holds first and counts against its limits.
func (t *tail) text() string {
	if t == nil {
		return ""
	}
	return strings.Join(t.texts, "")
}

// prependTo is chunk c spoken after the tail, or c when there is no tail: the tail's units, then c's. Its text in the
// static prompt starts at the tail's first unit, as a training row's text starts at its target's first word.
func (t *tail) prependTo(c chunk) chunk {
	if t == nil {
		return c
	}

	// The tail's last unit ended its chunk without a trailing space, so it gets one, as a unit inside a text has. The
	// tokenizer strips a unit's trailing space, so its tokens do not change.
	texts := slices.Concat(t.texts, c.texts)
	texts[len(t.texts)-1] += " "
	return chunk{units: slices.Concat(t.units, c.units), texts: texts, textUnits: len(t.units) + c.textUnits}
}
