package engine

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/taylorchu/2cent-tts/internal/fixtures"
	"github.com/taylorchu/2cent-tts/internal/text"
)

// TestUnitsGolden checks that written texts take the spoken form, and their units the tokens, training gave them.
func TestUnitsGolden(t *testing.T) {
	e, g := promptEngine(t), fixtures.LoadGolden(t)
	if len(g.Units) == 0 {
		t.Fatal("the golden file has no unit cases")
	}

	for _, c := range g.Units {
		spoken, err := text.Spoken(c.Text)
		if err != nil {
			t.Fatal(err)
		}

		if spoken != c.Spoken {
			t.Errorf("spoken form of %q = %q, want %q", c.Text, spoken, c.Spoken)
			continue
		}

		chunk, err := e.encodeChunk(spoken)
		if err != nil {
			t.Fatal(err)
		}

		if !slices.EqualFunc(chunk.units, c.Units, slices.Equal) {
			t.Errorf("units of %q = %v, want %v", spoken, chunk.units, c.Units)
		}
	}
}

// TestPrepareDropsUnknownTags speaks markup and sounds the model does not know as if they were not there.
func TestPrepareDropsUnknownTags(t *testing.T) {
	e := promptEngine(t)
	chunks := chunksOf(t, e, "Hi <dance> there.<br> <laugh>")
	if got := strings.Join(chunks[0].texts, ""); len(chunks) != 1 || got != "Hi there. <laugh>" {
		t.Errorf("%d chunks, the first %q, want one, %q", len(chunks), got, "Hi there. <laugh>")
	}
}

// TestPrepareRefusesBadText refuses text that holds no word or sound tag once written in its spoken form, and text
// longer than MaxTextCharacters.
func TestPrepareRefusesBadText(t *testing.T) {
	e := promptEngine(t)
	for _, written := range []string{"", "   ", "___", "(((( ...", "<br>", strings.Repeat("a ", MaxTextCharacters)} {
		if _, err := e.Prepare(written); !errors.Is(err, ErrBadText) {
			t.Errorf("Prepare(%q): %v, want a bad text", written, err)
		}
	}
}
