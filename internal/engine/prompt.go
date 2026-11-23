package engine

import (
	"slices"

	"github.com/taylorchu/2cent-tts/internal/tokenizer"
)

// markers are the token ids every prompt is built from. The prompts are the layout training wrote each row in, up to
// the target's first step, where generation picks up:
//
//	<s> <text> {reference text} <audio> {reference codes}    the voice's prompt, read once for each voice
//	<text> {the chunk's text}                                the chunk's static prompt, read once for each chunk
//	<audio> {the chunk's first LookaheadUnits units}         the target's opening
//
// Everything before the opening stays in view for the whole chunk. From the opening on, the main model sees only the
// window behind each position, as in training.
//
// A text is its units back to back, each encoded on its own, so a word has the same tokens wherever it sits. The
// unconditional prompt holds no reference and no text: <s> <text>, then the opening of a chunk with no unit, <audio>.
// A chunk that carries speech on is laid out the same way, the carried units first; tail.fill holds the rule.
type markers tokenizer.Markers

// voicePrompt is the prompt of a reference clip that says text and holds codes.
func (m markers) voicePrompt(text, codes []int) []int {
	ids := slices.Concat([]int{m.BOS, m.Text}, text, []int{m.Audio})
	for _, c := range codes {
		ids = append(ids, m.AudioBase+c)
	}
	return ids
}

// noVoicePrompt is the prompt of no reference clip.
func (m markers) noVoicePrompt() []int {
	return []int{m.BOS}
}

// unconditionalPrompt is the static prompt of the pass guidance compares against: no reference clip, and a chunk with
// no unit.
func (m markers) unconditionalPrompt() []int {
	return m.appendChunkStatic(m.noVoicePrompt(), chunk{})
}

// appendChunkStatic appends a chunk's static prompt to ids: its whole text under a <text> of its own, which a chunk
// with no unit holds empty.
func (m markers) appendChunkStatic(ids []int, c chunk) []int {
	return appendUnits(ids, m.Text, c.units[:c.textUnits])
}

// appendOpening appends a target's opening to ids: <audio>, then the opening units chunk.split gives.
func (m markers) appendOpening(ids []int, units [][]int) []int {
	return appendUnits(ids, m.Audio, units)
}

// appendUnits appends marker, then the units back to back, to ids.
func appendUnits(ids []int, marker int, units [][]int) []int {
	ids = append(ids, marker)
	for _, unit := range units {
		ids = append(ids, unit...)
	}
	return ids
}
