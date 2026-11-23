// Package text turns written text into what the model reads: the spoken form training saw, the chunks a text is
// spoken in, and the units each chunk is fed in. It counts tokens only through a function its caller passes, so it
// depends on the standard library alone.
package text

import (
	"fmt"
	"strings"
	"unicode"
)

// TagNamePattern is the regular expression of the name inside a sound tag's brackets, such as laugh in <laugh>.
const TagNamePattern = `[a-z_]+`

// tagPattern matches a sound tag.
var tagPattern = compile(`<` + TagNamePattern + `>`)

// FindTagIndex is where the first sound tag of s starts and ends, or nil when s holds none.
func FindTagIndex(s string) []int {
	if m, ok := tagPattern.find(s); ok {
		return m.index[:2]
	}
	return nil
}

// HasWord reports whether text holds a word, its sound tags aside.
func HasWord(text string) bool {
	return wordCharPattern.MatchString(tagPattern.ReplaceAllString(text, ""))
}

// TagEdges reports whether text opens with a sound tag and whether it closes with one, punctuation after it aside.
func TagEdges(text string) (opens, closes bool) {
	return openingTagPattern.MatchString(text), closingTagPattern.MatchString(text)
}

// spaceClass is every Unicode space, which the plain form writes as one space.
const spaceClass = `[\s\v\x{85}\p{Z}]`

var (
	dashRunPattern         = compile(`(?:` + spaceClass + `*-{2,})+`)
	dashPattern            = compile(`(?:` + spaceClass + `*-{2,})+` + spaceClass + `*`)
	spacesPattern          = compile(spaceClass + `+`)
	spaceBeforeMarkPattern = compile(spaceClass + `+([,.;:!?])`)
	// Python's \w: Unicode letters and digits, and the underscore.
	wordCharPattern     = compile(`[\p{L}\p{N}_]`)
	trailingTagsPattern = compile(`(?:` + spaceClass + `*<` + TagNamePattern + `>)*$`)
	openingTagPattern   = compile(`^` + spaceClass + `*<` + TagNamePattern + `>`)
	closingTagPattern   = compile(`<` + TagNamePattern + `>(?:` + spaceClass + `|[.,!?])*$`)
	accentBases         = accentBaseTable()
)

// accentBaseTable maps each character of decomposedGroups to the ASCII character it is written as.
func accentBaseTable() map[rune]byte {
	bases := map[rune]byte{}
	for _, group := range decomposedGroups {
		for _, r := range group[1:] {
			bases[r] = group[0]
		}
	}
	return bases
}

// Spoken is the text as training saw it: plain typography, and digits, symbols and abbreviations as words, its sound
// tags kept in place. Text that still holds a character the model has never seen is refused.
func Spoken(text string) (string, error) {
	out := spokenForm(text)
	unsaid := strings.Map(func(r rune) rune {
		if r < unicode.MaxASCII && (isASCIILetter(byte(r)) || strings.ContainsRune(spokenMarks, r)) {
			return -1
		}
		return r
	}, tagPattern.ReplaceAllString(out, ""))
	if unsaid != "" {
		return "", fmt.Errorf("text holds characters the model cannot say: %q", unsaid)
	}
	return out, nil
}

// spokenMarks are the only characters besides letters and tags that training texts hold.
const spokenMarks = "-.,!?;:'\"() "

func spokenForm(text string) string {
	// Trailing space says nothing, and the rules that end on $ read the text without it.
	text = strings.TrimRightFunc(text, unicode.IsSpace)
	// The written forms come before the plain form, which drops the underscores of a file name ("notes_v2.txt").
	out := betweenTags(text, func(stretch string) string {
		return withoutAccents(plainDashes(strings.ReplaceAll(writtenForms(stretch), "_", "")))
	})
	// Some transcripts set a space before punctuation ("is , according"), which no other text does.
	out = spaceBeforeMarkPattern.ReplaceAllString(spacesPattern.ReplaceAllString(out, " "), "$1")
	return strings.TrimSpace(out)
}

// betweenTags is text with each stretch between its sound tags rewritten, and each tag kept with a space on either
// side.
func betweenTags(text string, rewrite func(string) string) string {
	tags := tagPattern.FindAllString(text, -1)
	var b strings.Builder
	for i, stretch := range tagPattern.Split(text, -1) {
		if i > 0 {
			b.WriteString(" " + tags[i-1] + " ")
		}
		b.WriteString(rewrite(stretch))
	}
	return b.String()
}

// plainTypography is how every plain form writes r: a space for a tab, a line break or a typographic space, nothing
// for a soft hyphen or a zero-width character, and dashes, quotes and ligatures in ASCII. It reports false for a
// character it leaves as written.
func plainTypography(r rune) (string, bool) {
	switch {
	case typographySpace(r):
		return " ", true
	case strings.ContainsRune("\u00ad\u200b\u200c\u200d\u2060\ufeff", r):
		return "", true
	case r == '\u2014' || r == '\u2015' || r == '\u0097':
		return "--", true
	case r == '\u2010' || r == '\u2011' || r == '\u2013':
		return "-", true
	case strings.ContainsRune("\u201c\u201d\u201e\u201f", r):
		return `"`, true
	case strings.ContainsRune("`\u00b4\u2018\u2019\u201b\u02bc", r):
		return "'", true
	}
	plain, ok := ligatures[r]
	return plain, ok
}

// typographySpace is whether r is a tab, line break or typographic space, which plain text writes as a space.
func typographySpace(r rune) bool {
	return r == '\t' || r == '\n' || r == '\r' || r == '\u00a0' || (0x2000 <= r && r <= 0x200a) || r == '\u202f' ||
		r == '\u205f' || r == '\u3000'
}

var ligatures = map[rune]string{
	'\u2026': "...", '\u00e6': "ae", '\u00c6': "Ae", '\u0153': "oe", '\u0152': "Oe", '\u00df': "ss",
	'\ufb00': "ff", '\ufb01': "fi", '\ufb02': "fl", '\ufb03': "ffi", '\ufb04': "ffl", '\ufb05': "st", '\ufb06': "st",
}

// plainDashes sets every dash apart by spaces as " - ", and as " -" before a closing bracket, a closing quote or the
// end.
func plainDashes(text string) string {
	text = dashRunPattern.replaceWhere(text, func(m match) (string, bool) {
		return " -", dashClosesText(m.text[m.end():])
	})
	return dashPattern.ReplaceAllString(text, " - ")
}

func dashClosesText(rest string) bool {
	if strings.HasPrefix(rest, ")") || strings.TrimSpace(rest) == "" {
		return true
	}
	return (rest[0] == '"' || rest[0] == '\'') && (len(rest) == 1 || !isASCIILetter(rest[1]))
}

// withoutAccents writes each accented Latin letter as its base letter and drops a stray combining mark.
func withoutAccents(text string) string {
	return strings.Map(func(r rune) rune {
		if base, ok := accentBases[r]; ok {
			return rune(base)
		}

		if unicode.Is(unicode.Mn, r) {
			return -1
		}
		return r
	}, text)
}

// typographyForRules is the typography the training data's plain form also rewrites (every space, tab and line break as
// a space, soft hyphens and zero-width characters dropped, dashes, quotes and ligatures in ASCII), since the rules run
// first and match a literal space ("5\u2009kg", "Mr.\nBrown"); underscores stay, as file names need them. Subscript
// digits become digits ("H\u2082O").
func typographyForRules(text string) string {
	var b strings.Builder
	for _, r := range text {
		if 0x2080 <= r && r <= 0x2089 {
			b.WriteByte(byte('0' + r - 0x2080))
		} else if plain, ok := plainTypography(r); ok {
			b.WriteString(plain)
		} else if plain, ok := unitSigns[r]; ok {
			b.WriteString(plain)
		} else {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// unitSigns are the signs the written forms read in one form only: the micro sign, degree signs and the numero.
var unitSigns = map[rune]string{'\u03bc': "\u00b5", '\u2103': "\u00b0C", '\u2109': "\u00b0F", '\u2116': "No."}
