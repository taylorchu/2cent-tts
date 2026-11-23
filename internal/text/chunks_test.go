package text

import (
	"slices"
	"strings"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/fixtures"
)

// TestSentencesGolden checks that spoken texts split into the sentences training's split gives them.
func TestSentencesGolden(t *testing.T) {
	g := fixtures.LoadGolden(t)
	if len(g.Units) == 0 {
		t.Fatal("the golden file has no unit cases")
	}

	for _, c := range g.Units {
		if got := sentences(c.Spoken); !slices.Equal(got, c.Sentences) {
			t.Errorf("sentences(%q) = %q, want %q", c.Spoken, got, c.Sentences)
		}
	}
}

// testLimits are chunk limits counted in words, small enough for a test text to reach.
var testLimits = Limits{Tokens: 50, Sentences: 2, SentenceTokens: 100}

func words(s string) int {
	return len(strings.Fields(s))
}

// chunks packs text into chunks as Speak does when none carries speech on.
func chunks(text string, limits Limits, tokens func(string) int) []string {
	var out []string
	for pieces := Pieces(text, limits, tokens); len(pieces) > 0; {
		chunk, n := Pack(pieces, limits, tokens, "")
		out = append(out, chunk)
		pieces = pieces[n:]
	}
	return out
}

// TestUnits cuts texts into words and tags, each with the text up to the next, and finds none in punctuation alone.
func TestUnits(t *testing.T) {
	for text, want := range map[string][]string{
		"aa <laugh> bb cc.":  {"aa ", "<laugh> ", "bb ", "cc."},
		`"Hi," she said.`:    {`"Hi," `, "she ", "said."},
		"I can't. U.S. team": {"I ", "can't. ", "U.", "S. ", "team"},
		"<sigh>":             {"<sigh>"},
		"...":                nil,
		"":                   nil,
	} {
		if got := Units(text); !slices.Equal(got, want) {
			t.Errorf("Units(%q) = %q, want %q", text, got, want)
		}
	}
}

// TestSentences ends a sentence at a period, except after a title, an initial or an acronym's letter, and at the
// last period of an ellipsis.
func TestSentences(t *testing.T) {
	for text, want := range map[string][]string{
		"Mrs. Saunders left. J. K. Rowling stayed.": {"Mrs. Saunders left.", "J. K. Rowling stayed."},
		"The U.S. team won. We cheered!":            {"The U.S. team won.", "We cheered!"},
		"Well... I guess so. Fine.":                 {"Well... I guess so.", "Fine."},
		`He said "no." Then he left.`:               {`He said "no."`, "Then he left."},
		"I can't. My car won't start.":              {"I can't.", "My car won't start."},
		"You got straight A's. Congratulations!":    {"You got straight A's.", "Congratulations!"},
		"It was the dogs'. They left.":              {"It was the dogs'.", "They left."},
		"Ask for Dr.":                               {"Ask for Dr."},
		"Hi.Bye.":                                   {"Hi.Bye."},
	} {
		if got := sentences(text); !slices.Equal(got, want) {
			t.Errorf("sentences(%q) = %q, want %q", text, got, want)
		}
	}
}

// TestPackFillsChunksToTheirLimits packs sentences into chunks up to the sentence and token limits, closing the last
// with a period.
func TestPackFillsChunksToTheirLimits(t *testing.T) {
	got := chunks("One. Two? \"Three!\" Four", testLimits, words)
	if !slices.Equal(got, []string{"One. Two?", "\"Three!\" Four."}) {
		t.Errorf("short sentences were not packed %d to a chunk: %q", testLimits.Sentences, got)
	}

	for _, chunk := range chunks(strings.Repeat("Word word. ", 60), testLimits, words) {
		if n := words(chunk); n > testLimits.Tokens {
			t.Errorf("chunk of %d tokens", n)
		}
	}
}

// TestPiecesSplitOnlySentencesOverTheLimit keeps a sentence within SentenceTokens whole and splits a longer one into
// parts within it.
func TestPiecesSplitOnlySentencesOverTheLimit(t *testing.T) {
	sentence := strings.Repeat("a clause of words, ", 20) + "end."
	got := chunks("Hi. "+sentence+" Bye.", testLimits, words)
	if !slices.Equal(got, []string{"Hi.", sentence, "Bye."}) {
		t.Errorf("a sentence within SentenceTokens was not one chunk: %q", got)
	}

	// After its commas, or, in unpunctuated text, after a word, so no chunk runs past the length cap.
	for _, text := range []string{strings.Repeat("a clause of words, ", 40) + "end.",
		strings.Repeat("and then we walked ", 60) + "home."} {
		checkSplitWhole(t, text, " ", words)
	}

	first := chunks(strings.Repeat("a clause of words, ", 40)+"end.", testLimits, words)[0]
	if !strings.HasSuffix(first, ",") {
		t.Errorf("a clause was split inside: %q", first)
	}

	// A word with no space in it splits between its units, so a run of digits or symbols is spoken whole too.
	characters := func(s string) int { return len(s) / 4 }
	glued := strings.Repeat("one,", 500) + "end."
	checkSplitWhole(t, glued, "", characters)
}

// checkSplitWhole checks that text packs into two or more chunks, each within SentenceTokens, that join back to it.
func checkSplitWhole(t *testing.T, text, separator string, tokens func(string) int) {
	t.Helper()
	got := chunks(text, testLimits, tokens)
	if len(got) < 2 || strings.Join(got, separator) != text {
		t.Errorf("%.30q was not split whole: %d chunks", text, len(got))
	}

	for _, chunk := range got {
		if tokens(chunk) > testLimits.SentenceTokens {
			t.Errorf("a chunk of %d tokens", tokens(chunk))
		}
	}
}

// TestTagsStayWithTheirSentence keeps a sound tag after a sentence end in that sentence's chunk.
func TestTagsStayWithTheirSentence(t *testing.T) {
	for text, want := range map[string][]string{
		"That is so funny. <laugh>":    {"That is so funny. <laugh>"},
		"Hi there. Okay then. <laugh>": {"Hi there. Okay then. <laugh>"},
		"Hello there <sigh>":           {"Hello there. <sigh>"},
		"Hi. <laugh> Okay.":            {"Hi. <laugh> Okay."},
	} {
		if got := chunks(text, testLimits, words); !slices.Equal(got, want) {
			t.Errorf("chunks(%q) = %q, want %q", text, got, want)
		}
	}
}

// TestPackCountsCarriedText packs after text a chunk carries on: its sentences and tokens count against the limits,
// and the first piece goes in whatever they say.
func TestPackCountsCarriedText(t *testing.T) {
	limits := Limits{Tokens: 8, Sentences: 3, SentenceTokens: 100}
	pieces := Pieces("One two. Three four. Five six.", limits, words)
	for _, tc := range []struct {
		carried string
		want    string
		n       int
	}{
		{"", "One two. Three four. Five six.", 3},
		{"slower models.", "One two. Three four.", 2},
		{"Yes. and then slower models.", "One two.", 1},
		{"a b c d e", "One two.", 1},
	} {
		got, n := Pack(pieces, limits, words, tc.carried)
		if got != tc.want || n != tc.n {
			t.Errorf("after %q: %q of %d pieces, want %q of %d", tc.carried, got, n, tc.want, tc.n)
		}
	}
}

// TestEveryPieceSaysSomething splits a sentence over the limit whose last clause is punctuation alone, and one whose
// first clause is: no piece is left with no word, since a chunk after carried speech may hold that piece alone.
func TestEveryPieceSaysSomething(t *testing.T) {
	limits := Limits{Tokens: 8, Sentences: 2, SentenceTokens: 4}
	dashes := strings.Repeat("- ", 6) + "-"
	for _, text := range []string{
		"Hi there. one two three, four five six, " + dashes + ". Bye now.",
		dashes + ", one two three, four five six. Bye now.",
	} {
		var all []string
		for _, piece := range Pieces(text, limits, words) {
			if Units(piece.text) == nil {
				t.Errorf("%q: a piece %q holds no word", text, piece.text)
			}
			all = append(all, piece.text)
		}
		if got := strings.Join(all, " "); got != strings.TrimSpace(text) {
			t.Errorf("%q: pieces %q lost or moved text", text, all)
		}
	}
}
