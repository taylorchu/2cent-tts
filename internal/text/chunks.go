package text

import (
	"slices"
	"strings"
)

// wordPattern is a word as the training data's word timings count one: letters, with apostrophes inside. The text it
// reads is spoken, whose letters are ASCII.
var wordPattern = compile(`[A-Za-z]+(?:'[A-Za-z]+)*`)

// Units cuts a chunk into the units it is fed in, a word or a sound tag with the text up to the next, by the text
// alone, as training cut them. The first unit also holds what the text opens with. Text with no word or tag, such as
// punctuation alone, holds no unit: it says nothing.
func Units(text string) []string {
	blank := tagPattern.replaceFunc(text, func(tag match) string { return strings.Repeat(" ", tag.end()-tag.start()) })
	var starts []int
	for _, word := range wordPattern.findAll(blank) {
		starts = append(starts, word.start())
	}

	for _, tag := range tagPattern.findAll(text) {
		starts = append(starts, tag.start())
	}

	if len(starts) == 0 {
		return nil
	}

	slices.Sort(starts)
	edges := slices.Concat([]int{0}, starts[1:], []int{len(text)})
	units := make([]string, len(edges)-1)
	for i := range units {
		units[i] = text[edges[i]:edges[i+1]]
	}
	return units
}

// Limits bound a chunk: at most Tokens text tokens and Sentences sentences, each part of a split sentence counting as
// a sentence of its own.
type Limits struct {
	Tokens, Sentences int
	// A sentence up to SentenceTokens is one chunk, whatever Tokens says: training targets hold sentences this long
	// and always end on a sentence end. Only a longer one splits, after its commas, or after a word where a clause
	// alone runs longer, with no period added there.
	SentenceTokens int
}

// Piece is a sentence, or a part of one over SentenceTokens, and whether it ends its sentence.
type Piece struct {
	text string
	ends bool
}

// Pieces cuts text into the pieces chunks are packed from: its sentences, a sentence with no words joined to the one
// before, and a sentence over limits.SentenceTokens split into parts no longer. Every piece of a text that says
// something holds a word or a sound tag, since a chunk after carried speech may hold one piece alone.
func Pieces(text string, limits Limits, tokens func(string) int) []Piece {
	var pieces []Piece
	for _, sentence := range sentences(text) {
		// A sentence with no words, such as a trailing tag, joins the sentence before, as training text keeps it.
		if len(pieces) > 0 && !HasWord(sentence) {
			pieces[len(pieces)-1].text += " " + sentence
			continue
		}

		if tokens(sentence) <= limits.SentenceTokens {
			pieces = append(pieces, Piece{sentence, true})
			continue
		}

		split := splitSentence(sentence, limits.SentenceTokens, tokens)
		for i, part := range split {
			pieces = append(pieces, Piece{part, i == len(split)-1})
		}
	}
	return joinUnitless(pieces)
}

// joinUnitless joins each piece with no word or sound tag, such as a run of dashes a long sentence split off, onto the
// piece before it, or onto the piece after it when it comes first.
func joinUnitless(pieces []Piece) []Piece {
	var joined []Piece
	leading := ""
	for _, piece := range pieces {
		switch {
		case Units(piece.text) != nil:
			piece.text = strings.TrimSpace(leading + " " + piece.text)
			leading = ""
			joined = append(joined, piece)
		case len(joined) > 0:
			last := &joined[len(joined)-1]
			last.text, last.ends = last.text+" "+piece.text, piece.ends
		default:
			leading = strings.TrimSpace(leading + " " + piece.text)
		}
	}

	if leading != "" {
		joined = append(joined, Piece{leading, true})
	}
	return joined
}

// splitSentence cuts a sentence over limit tokens into parts of up to limit tokens: after its commas, then after a
// word within a clause that alone runs over limit, as unpunctuated chat and transcripts do, then between the units of
// a word that alone runs over limit, as digits or symbols with no space between them do.
func splitSentence(sentence string, limit int, tokens func(string) int) []string {
	var parts []string
	part := ""
	add := func(piece, space string) {
		joined := strings.TrimSpace(part + space + piece)
		if part != "" && tokens(joined) > limit {
			parts = append(parts, part)
			joined = piece
		}
		part = joined
	}

	for _, clause := range strings.SplitAfter(sentence, ", ") {
		clause = strings.TrimSpace(clause)
		if tokens(clause) <= limit {
			add(clause, " ")
			continue
		}

		for word := range strings.FieldsSeq(clause) {
			units := Units(word)
			if tokens(word) <= limit || len(units) < 2 {
				add(word, " ")
				continue
			}

			for i, unit := range units {
				space := ""
				if i == 0 {
					space = " "
				}
				add(unit, space)
			}
		}
	}
	return append(parts, part)
}

// Pack is the next chunk of pieces within limits, and how many pieces it holds. carried is text the chunk's speech
// continues, which stands before it in the chunk's text and counts against limits, its sentences and tokens both, so a
// chunk that carries speech on is no longer than a training target. The first piece goes in whatever the limits say.
func Pack(pieces []Piece, limits Limits, tokens func(string) int, carried string) (chunk string, count int) {
	current, sentenceCount, pieceCount := Piece{ends: true}, len(sentences(carried)), 0
	for _, piece := range pieces {
		joined := strings.TrimSpace(current.text + " " + piece.text)
		full := sentenceCount >= limits.Sentences
		if current.text != "" && (full || tokens(strings.TrimSpace(carried+" "+joined)) > limits.Tokens) {
			break
		}
		current, sentenceCount, pieceCount = Piece{joined, piece.ends}, sentenceCount+1, pieceCount+1
	}

	if current.ends {
		return closed(current.text), pieceCount
	}
	return current.text, pieceCount
}

// closed ends the chunk on a sentence end, before any sound tags that close it. Every training text ends on
// punctuation, or on sound tags after it, so every chunk does too, except one that ends where a sentence was split.
func closed(chunk string) string {
	tags, _ := trailingTagsPattern.find(chunk)
	tail := tags.start()
	body := chunk[:tail]
	if body == "" || endsSentence(body) {
		return chunk
	}
	return strings.TrimRight(body, ",;:") + "." + chunk[tail:]
}

// endsSentence reports whether text ends on ".", "!" or "?", closing quotes and brackets aside.
func endsSentence(text string) bool {
	text = strings.TrimRight(text, `"')`)
	return text != "" && strings.ContainsAny(text[len(text)-1:], ".!?")
}

// continuingAbbreviations are the words whose period takes more after it ("Mrs. Saunders", "vs. them").
var continuingAbbreviations = map[string]bool{"Mr": true, "Mrs": true, "Ms": true, "Dr": true, "St": true, "Jr": true,
	"Sr": true, "vs": true, "etc": true}

// periodContinuesSentence is whether the period that closes body ends no sentence: after a title or word that takes
// more after it, after a letter standing alone (an initial or a dotted acronym: "J. K.", "U.S."; a contraction's last
// letter is no such letter: "I can't.") and as the last period of an ellipsis. Speech goes on past each, so a chunk
// break there would fall mid-phrase.
func periodContinuesSentence(body string) bool {
	body = strings.TrimRight(body, `"')`)
	if !strings.HasSuffix(body, ".") {
		return false
	}

	word := body[:len(body)-1]
	if strings.HasSuffix(word, "..") {
		return true
	}

	start := len(word)
	for start > 0 && isASCIILetter(word[start-1]) {
		start--
	}

	run := word[start:]
	if run == "" {
		return false
	}

	// What stands before the letters: a digit or an underscore makes them part of a longer word.
	inWord := start > 0 && (word[start-1] == '_' || isASCIIDigit(word[start-1]))
	if len(run) == 1 {
		return !inWord && (start == 0 || word[start-1] != '\'')
	}
	return !inWord && continuingAbbreviations[run]
}

// sentences cuts text after each ".", "!" or "?" that a space or the end follows, closing quotes and brackets staying
// with their sentence. A period that ends no sentence (periodContinuesSentence) is passed over.
func sentences(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); i++ {
		if !strings.ContainsRune(".!?", rune(text[i])) {
			continue
		}

		end := i + 1
		for end < len(text) && strings.ContainsRune(`"')`, rune(text[end])) {
			end++
		}

		if (end == len(text) || text[end] == ' ') && !periodContinuesSentence(text[start:end]) {
			if s := strings.TrimSpace(text[start:end]); s != "" {
				out = append(out, s)
			}
			start, i = end, end
		}
	}

	if rest := strings.TrimSpace(text[start:]); rest != "" {
		out = append(out, rest)
	}
	return out
}
