package tokenizer

import (
	"slices"
	"strings"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

func releaseTokenizer(tb testing.TB) *Tokenizer {
	tb.Helper()
	textTokenizer, err := New([]byte(fixtures.LoadWeights(tb).Metadata["tokenizer"]))
	if err != nil {
		tb.Fatal(err)
	}

	return textTokenizer
}

// TestTokenizer encodes golden texts to the ids the training tokenizer gives.
func TestTokenizer(t *testing.T) {
	textTokenizer, g := releaseTokenizer(t), fixtures.LoadGolden(t)
	if len(g.Tokenizer) == 0 {
		t.Fatal("the golden file holds no tokenizer cases")
	}

	for _, c := range g.Tokenizer {
		ids, err := textTokenizer.Encode(c.Text)
		if err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(ids, c.IDs) {
			t.Errorf("Encode(%q) = %v, want %v", c.Text, ids, c.IDs)
		}
	}
}

// TestTokenizerSplitsContractions encodes each English contraction ending as GPT-2's pre-tokenizer splits it: one
// token of its own after the word it ends. An ending split as punctuation and letters takes two.
func TestTokenizerSplitsContractions(t *testing.T) {
	textTokenizer := releaseTokenizer(t)
	word, err := textTokenizer.Encode("they")
	if err != nil {
		t.Fatal(err)
	}

	for _, ending := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
		whole, err := textTokenizer.Encode("they" + ending)
		if err != nil {
			t.Fatal(err)
		}

		if len(whole) != len(word)+1 || !slices.Equal(whole[:len(word)], word) {
			t.Errorf("%q: %v, want %v and one token", "they"+ending, whole, word)
		}
	}
}

// TestTokenizerRefusesControlTags refuses the prompt's markers and tags that are no sound the model knows.
func TestTokenizerRefusesControlTags(t *testing.T) {
	textTokenizer := releaseTokenizer(t)
	for _, text := range []string{"a <s> b", "a <audio> b", "a <text> b", "a <pad> b", "a <dance> b"} {
		if ids, err := textTokenizer.Encode(text); err == nil {
			t.Errorf("Encode(%q) = %v, want an error", text, ids)
		}
	}
}

// TestDropUnknownTags drops tags that are no sound the model knows, keeps sound tags and text that is no tag, and
// leaves a text Encode takes.
func TestDropUnknownTags(t *testing.T) {
	textTokenizer := releaseTokenizer(t)
	for text, want := range map[string]string{
		"Use <br> here.":         "Use   here.",
		"<b>bold</b> <laugh> ok": " bold  <laugh> ok",
		"a <audio> b <dance> c":  "a   b   c",
		"I <3 you, a<b, c>d.":    "I <3 you, a<b, c>d.",
	} {
		if got := textTokenizer.DropUnknownTags(text); got != want {
			t.Errorf("DropUnknownTags(%q) = %q, want %q", text, got, want)
		}

		if _, err := textTokenizer.Encode(textTokenizer.DropUnknownTags(text)); err != nil {
			t.Errorf("%q still holds a tag Encode refuses: %v", text, err)
		}
	}
}

// TestNewCountsAudioCodes counts the audio code tokens and refuses codes whose ids have a gap.
func TestNewCountsAudioCodes(t *testing.T) {
	spec := `{"added_tokens": [{"id": 0, "content": "<s>"}, {"id": 1, "content": "</s>"},
		{"id": 2, "content": "<text>"}, {"id": 3, "content": "<audio>"}, {"id": 4, "content": "<audio_0>"},
		{"id": 5, "content": "<audio_1>"}, {"id": 6, "content": "<laugh>"}],
		"normalizer": ` + releaseNormalizer + `, "pre_tokenizer": ` + releasePreTokenizer + `,
		"model": {"type": "BPE", "vocab": {}, "merges": []}}`
	textTokenizer, err := New([]byte(spec))
	if err != nil {
		t.Fatal(err)
	}

	ids, err := textTokenizer.Encode("<laugh>")
	if textTokenizer.AudioCodeCount() != 2 || err != nil || !slices.Equal(ids, []int{6}) {
		t.Errorf("%d audio codes, <laugh> encoded as %v, error %v", textTokenizer.AudioCodeCount(), ids, err)
	}

	if _, err := New([]byte(strings.Replace(spec, `"id": 5`, `"id": 7`, 1))); err == nil {
		t.Error("audio codes with a gap in their ids were accepted")
	}

	// Only the release's normalizer and pre-tokenizer are implemented, so any other is refused, not misread.
	if _, err := New([]byte(strings.Replace(spec, `"use_regex":true`, `"use_regex":false`, 1))); err == nil {
		t.Error("a pre-tokenizer the tokenizer does not implement was accepted")
	}
}
