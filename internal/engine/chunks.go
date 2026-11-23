package engine

import (
	"errors"
	"fmt"
	"slices"
	"unicode/utf8"

	"github.com/taylorchu/2cent-tts/internal/text"
)

// chunk is the text spoken from one fresh copy of the voice's prompt: each unit's tokens, encoded apart, then
// LookaheadUnits units of </s>. A chunk with no unit holds no </s> either.
type chunk struct {
	units [][]int
	// texts is each unit's spoken text, for the text a later chunk carries on.
	texts []string
	// textUnits is how many units are the text's; the </s> units that close it follow.
	textUnits int
}

// textTokens is the chunk's text, its units back to back.
func (c chunk) textTokens() []int {
	return slices.Concat(c.units[:c.textUnits]...)
}

// split is the chunk's units in its target's opening, the first lookahead of them, and the ones fed later.
func (c chunk) split(lookahead int) (opening, later [][]int) {
	open := min(lookahead, len(c.units))
	return c.units[:open], c.units[open:]
}

// Script is written text checked for Speak: its spoken form, cut into pieces that encode, each a sentence or a part of
// a long one. Speak packs them into chunks as it goes, since a chunk that carries speech on holds that speech's text
// too.
type Script struct {
	pieces []text.Piece
}

// ErrBadText marks Prepare's refusals of a text, which its caller can fix.
var ErrBadText = errors.New("bad text")

// MaxTextCharacters is the longest text Prepare takes, as OpenAI's speech API takes.
const MaxTextCharacters = 4096

// Prepare checks written text and cuts it into pieces, so a text the model cannot say is refused before any audio is
// made. Tags that name no sound the model knows are dropped.
func (e *Engine) Prepare(written string) (Script, error) {
	if n := utf8.RuneCountInString(written); n > MaxTextCharacters {
		return Script{}, fmt.Errorf("%w: the text is %d characters long, more than %d", ErrBadText, n,
			MaxTextCharacters)
	}

	spoken, err := text.Spoken(e.tokenizer.DropUnknownTags(written))
	if err != nil {
		return Script{}, fmt.Errorf("%w: %w", ErrBadText, err)
	}

	script := Script{pieces: text.Pieces(spoken, e.settings.limits(), e.tokenCount)}
	if len(script.pieces) == 0 {
		return Script{}, errNothingToSay
	}

	// Every piece holds a word or a sound tag and encodes the same in any chunk, so a chunk packed after carried speech
	// encodes whenever the pieces packed alone do.
	_, err = e.packAll(script.pieces)
	if err != nil {
		return Script{}, err
	}
	return script, nil
}

// packAll is every chunk the pieces pack into when no chunk carries speech on.
func (e *Engine) packAll(pieces []text.Piece) ([]chunk, error) {
	var chunks []chunk
	for len(pieces) > 0 {
		next, n, err := e.pack(pieces, "")
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, next)
		pieces = pieces[n:]
	}
	return chunks, nil
}

// errNothingToSay refuses a text that holds no word or sound tag.
var errNothingToSay = fmt.Errorf("%w: the text holds nothing to say", ErrBadText)

// pack is the next chunk of pieces after the carried text, and how many pieces it holds. A chunk holds at least one
// word or sound tag, as every training target did.
func (e *Engine) pack(pieces []text.Piece, carried string) (chunk, int, error) {
	packed, n := text.Pack(pieces, e.settings.limits(), e.tokenCount, carried)
	c, err := e.encodeChunk(packed)
	if err != nil {
		return chunk{}, n, fmt.Errorf("%w: %w", ErrBadText, err)
	}

	if c.textUnits == 0 {
		return chunk{}, n, errNothingToSay
	}
	return c, n, nil
}

// nextChunk is the chunk Speak speaks next from the pending pieces, after the carried tail, if any, and the pieces
// left. The tail's text stands first in the chunk and counts against its limits.
func (e *Engine) nextChunk(pending []text.Piece, carried *tail) (chunk, []text.Piece, error) {
	next, n, err := e.pack(pending, carried.text())
	if err != nil {
		return chunk{}, nil, err
	}
	return carried.prependTo(next), pending[n:], nil
}

// tokenCount is how many tokens a text encodes to. A text the tokenizer cannot encode fails when its units are
// encoded, so it counts as none here.
func (e *Engine) tokenCount(s string) int {
	ids, _ := e.tokenizer.Encode(s)
	return len(ids)
}

// encodeChunk is the tokens of each unit of a chunk, encoded apart so a unit's tokens are the same wherever it is fed,
// then LookaheadUnits units of </s>. A start feeds the unit LookaheadUnits ahead, and what is fed is the model's only
// sign that its start was counted, so the starts of the last words must feed something too.
func (e *Engine) encodeChunk(spoken string) (chunk, error) {
	var c chunk
	for _, unit := range text.Units(spoken) {
		ids, err := e.tokenizer.Encode(unit)
		if err != nil {
			return chunk{}, err
		}
		c.units = append(c.units, ids)
		c.texts = append(c.texts, unit)
	}

	c.textUnits = len(c.units)
	if c.textUnits == 0 {
		return chunk{}, nil
	}

	for range e.settings.LookaheadUnits {
		c.units = append(c.units, []int{e.markers.EOS})
	}
	return c, nil
}
