package text

import (
	"testing"
)

// TestTagEdges finds a sound tag that opens a text or closes it, punctuation and spaces aside.
func TestTagEdges(t *testing.T) {
	for text, want := range map[string][2]bool{
		"For an evening out.":          {false, false},
		"<laugh> For an evening out.":  {true, false},
		"For an evening out. <sigh>.":  {false, true},
		"For a <laugh> evening out.":   {false, false},
		" <laugh> Hi there <breath> !": {true, true},
	} {
		if opens, closes := TagEdges(text); [2]bool{opens, closes} != want {
			t.Errorf("TagEdges(%q) = %v, %v, want %v", text, opens, closes, want)
		}
	}
}
