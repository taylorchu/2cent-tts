// Package tokenizer is the byte-level BPE the model was trained with, read from the tokenizer.json the release file
// holds. It knows sound tags from the text package and nothing of the model or the codec.
package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"github.com/taylorchu/2cent-tts/internal/text"
)

// Tokenizer is the byte-level BPE of a Hugging Face tokenizer.json, with its added tokens kept whole. It is only read
// once made, so any number of goroutines may share it.
type Tokenizer struct {
	vocab      map[string]int
	ranks      map[[2]string]int
	added      map[string]int
	byteRunes  [256]rune
	audioCodes int
	markers    Markers
}

// Markers are the token ids the prompt is laid out with.
type Markers struct {
	// BOS is the id of <s>, which opens a prompt.
	BOS int
	// Text and Audio are the ids of <text> and <audio>, which open a prompt's text and a target's speech.
	Text, Audio int
	// EOS is the id of </s>, which closes a text.
	EOS int
	// AudioBase is the id of <audio_0>, the first audio code.
	AudioBase int
}

// Markers is the tokenizer's marker ids.
func (t *Tokenizer) Markers() Markers {
	return t.markers
}

type tokenizerSpec struct {
	AddedTokens  []addedTokenSpec `json:"added_tokens"`
	Normalizer   json.RawMessage  `json:"normalizer"`
	PreTokenizer json.RawMessage  `json:"pre_tokenizer"`
	Model        struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges [][2]string    `json:"merges"`
	} `json:"model"`
}

type addedTokenSpec struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
}

// The release's normalizer and pre-tokenizer, which normalize and splitWords implement; any other is refused.
const (
	releaseNormalizer = `{"type":"Sequence","normalizers":[{"type":"Strip","strip_left":true,"strip_right":true},` +
		`{"type":"Prepend","prepend":" "},{"type":"Replace","pattern":{"String":" ,"},"content":","},` +
		`{"type":"Replace","pattern":{"String":" ."},"content":"."},{"type":"Replace","pattern":{"String":" !"},` +
		`"content":"!"},{"type":"Replace","pattern":{"String":" ?"},"content":"?"},{"type":"Replace","pattern":` +
		`{"String":" ;"},"content":";"},{"type":"Replace","pattern":{"String":" :"},"content":":"}]}`
	releasePreTokenizer = `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}`
)

var spaceBeforeMark = strings.NewReplacer(" ,", ",", " .", ".", " !", "!", " ?", "?", " ;", ";", " :", ":")

// normalize applies the release's normalizer: s trimmed, a space put before it, and no space before a mark.
func normalize(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	return spaceBeforeMark.Replace(" " + s)
}

// sameJSON reports whether raw and want hold the same JSON value.
func sameJSON(raw json.RawMessage, want string) bool {
	var got, expected any
	return json.Unmarshal(raw, &got) == nil && json.Unmarshal([]byte(want), &expected) == nil &&
		reflect.DeepEqual(got, expected)
}

// New reads a tokenizer.json whose audio codes are the added tokens <audio_0>, <audio_1> and on, with consecutive ids.
func New(data []byte) (*Tokenizer, error) {
	var spec tokenizerSpec
	err := json.Unmarshal(data, &spec)
	if err != nil {
		return nil, err
	}

	if spec.Model.Type != "BPE" || !sameJSON(spec.Normalizer, releaseNormalizer) ||
		!sameJSON(spec.PreTokenizer, releasePreTokenizer) {
		return nil, errors.New("unsupported tokenizer layout")
	}

	ranks := map[[2]string]int{}
	for i, pair := range spec.Model.Merges {
		ranks[pair] = i
	}

	t := &Tokenizer{vocab: spec.Model.Vocab, ranks: ranks, added: map[string]int{}, byteRunes: byteLevelRunes()}

	for _, a := range spec.AddedTokens {
		if !strings.HasPrefix(a.Content, "<") || !strings.HasSuffix(a.Content, ">") {
			return nil, fmt.Errorf("added token %q is not a <tag>", a.Content)
		}
		t.added[a.Content] = a.ID
	}

	err = t.findMarkers()
	if err != nil {
		return nil, err
	}
	return t, nil
}

// findMarkers sets the prompt's marker ids from the added tokens and counts the audio codes, whose ids must follow
// one another from AudioBase.
func (t *Tokenizer) findMarkers() error {
	ids := &t.markers
	for _, m := range []struct {
		id    *int
		token string
	}{
		{&ids.BOS, "<s>"}, {&ids.Text, "<text>"}, {&ids.Audio, "<audio>"}, {&ids.EOS, "</s>"},
		{&ids.AudioBase, "<audio_0>"},
	} {
		id, ok := t.added[m.token]
		if !ok {
			return fmt.Errorf("tokenizer is missing %s", m.token)
		}
		*m.id = id
	}

	// Audio ids are computed as base + code, so the block must be contiguous.
	for {
		id, ok := t.added[fmt.Sprintf("<audio_%d>", t.audioCodes)]
		if !ok {
			return nil
		}

		if id != t.markers.AudioBase+t.audioCodes {
			return errors.New("audio token ids are not contiguous")
		}
		t.audioCodes++
	}
}

// AudioCodeCount is how many audio codes the tokenizer holds.
func (t *Tokenizer) AudioCodeCount() int {
	return t.audioCodes
}

// byteLevelRunes is GPT-2's byte to character table: printable bytes stand for themselves, the rest move past 255.
func byteLevelRunes() [256]rune {
	var runes [256]rune
	next := 256
	for b := range 256 {
		if (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff) {
			runes[b] = rune(b)
		} else {
			runes[b] = rune(next)
			next++
		}
	}
	return runes
}

// TokenCount is one more than the largest token id: how many ids the tokenizer can give, the audio codes included.
func (t *Tokenizer) TokenCount() int {
	n := 0
	for _, ids := range []map[string]int{t.vocab, t.added} {
		for _, id := range ids {
			n = max(n, id+1)
		}
	}
	return n
}

// markupPattern matches a <word> or a </word>: the shape of a sound tag, and of the markup chat and web text holds.
var markupPattern = regexp.MustCompile(`</?` + text.TagNamePattern + `>`)

// DropUnknownTags removes every <word> and </word> of s that is not a sound tag the model knows, such as markup
// (<br>, <b>) or a sound no training text held, leaving a space; the words around it are read as written.
func (t *Tokenizer) DropUnknownTags(s string) string {
	return markupPattern.ReplaceAllStringFunc(s, func(tag string) string {
		if _, ok := t.soundTag(tag); ok {
			return tag
		}
		return " "
	})
}

// soundTag is the id of a sound tag the model knows. Sound tags follow the audio tokens; added tokens before them
// mark the prompt's structure.
func (t *Tokenizer) soundTag(tag string) (int, bool) {
	id, ok := t.added[tag]
	return id, ok && id >= t.markers.AudioBase+t.audioCodes
}

// Encode tokenizes s the way the trainer did: tags whole, and each stretch between tags normalized on its own.
func (t *Tokenizer) Encode(s string) ([]int, error) {
	var ids []int
	for s != "" {
		loc := text.FindTagIndex(s)
		end := len(s)
		if loc != nil {
			end = loc[0]
		}

		piece := normalize(s[:end])

		for _, word := range splitWords(piece) {
			wordIDs, err := t.bpe(word)
			if err != nil {
				return nil, err
			}
			ids = append(ids, wordIDs...)
		}

		if loc == nil {
			break
		}

		tag := s[loc[0]:loc[1]]
		id, ok := t.soundTag(tag)
		if !ok {
			return nil, fmt.Errorf("%s is not a sound tag the model knows", tag)
		}
		ids = append(ids, id)
		s = s[loc[1]:]
	}
	return ids, nil
}

// bpe is the ids of one pre-tokenized word: its bytes as symbols, merged a pair at a time in the merges' rank order.
func (t *Tokenizer) bpe(word string) ([]int, error) {
	var symbols []string
	for _, b := range []byte(word) {
		symbols = append(symbols, string(t.byteRunes[b]))
	}

	for len(symbols) > 1 {
		best, at := -1, -1
		for i := 0; i+1 < len(symbols); i++ {
			rank, ok := t.ranks[[2]string{symbols[i], symbols[i+1]}]
			if ok && (best < 0 || rank < best) {
				best, at = rank, i
			}
		}

		if at < 0 {
			break
		}

		first, second := symbols[at], symbols[at+1]
		// The pair at is the first of its rank, so no earlier pair is the same.
		for i := at; i+1 < len(symbols); i++ {
			if symbols[i] == first && symbols[i+1] == second {
				symbols[i] = first + second
				symbols = slices.Delete(symbols, i+1, i+2)
			}
		}
	}

	ids := make([]int, len(symbols))
	for i, s := range symbols {
		id, ok := t.vocab[s]
		if !ok {
			return nil, fmt.Errorf("%q has no token", word)
		}
		ids[i] = id
	}
	return ids, nil
}

// splitWords is the GPT-2 pre-tokenizer pattern
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// written out, since Go's regexp has no lookahead.
func splitWords(s string) []string {
	r := []rune(s)
	var words []string
	isOther := func(c rune) bool { return !unicode.IsSpace(c) && !unicode.IsLetter(c) && !unicode.IsNumber(c) }
	run := func(i int, class func(rune) bool) int {
		for i < len(r) && class(r[i]) {
			i++
		}
		return i
	}

	for i := 0; i < len(r); {
		end := contractionEnd(r, i)
		if end == i {
			j := i
			if r[j] == ' ' && j+1 < len(r) {
				j++
			}

			for _, class := range []func(rune) bool{unicode.IsLetter, unicode.IsNumber, isOther} {
				if class(r[j]) {
					end = run(j, class)
					break
				}
			}
		}

		if end == i {
			end = run(i, unicode.IsSpace)
			if end < len(r) && end-i > 1 {
				end--
			}
		}

		words = append(words, string(r[i:end]))
		i = end
	}
	return words
}

// contractionEnd is where the contraction starting at i ends ("'s", "'ll"), or i where none starts there.
func contractionEnd(r []rune, i int) int {
	if r[i] != '\'' {
		return i
	}

	for _, c := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
		if strings.HasPrefix(string(r[i+1:min(len(r), i+3)]), c) {
			return i + 1 + len(c)
		}
	}
	return i
}
