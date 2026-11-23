package text

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// sentenceStarters are capitalized words that start a sentence, so "Main St. The house" is a street, not "Saint The".
var sentenceStarters = set("The", "Then", "It", "He", "She", "We", "They", "I", "A", "An", "This", "That", "There",
	"These", "Those", "But", "And", "So", "In", "On", "At", "His", "Her", "My", "Our", "Their", "When", "If", "As",
	"After", "Before", "Your")

// wordKind is the kind of word nextWord tells apart.
type wordKind int

const (
	wordNone wordKind = iota
	wordQuoted
	wordInitial
	wordLetter
	wordStarter
	wordCapitals
	wordName
	wordOther
)

// sentenceStartKinds are the kinds of word after a period that show the period ended its sentence; a lowercase word
// or anything else after it continues the sentence.
var sentenceStartKinds = map[wordKind]bool{wordNone: true, wordQuoted: true, wordInitial: true,
	wordLetter: true, wordStarter: true, wordCapitals: true, wordName: true}

var (
	closingPattern           = compile(`^[\"'\x{201d}\x{2019})\]]{0,3}`)
	openingBracketPattern    = compile(`^[\"'\x{201d}\x{2019})\]]{0,3}\s+[(\[]([A-Za-z]*)`)
	quotedPattern            = compile(`^[\"'\x{201c}\x{2018}][A-Z]`)
	initialPattern           = compile(`^[A-Z]\.`)
	leadingLettersPattern    = compile(`^[A-Za-z]+`)
	wordBeforePattern        = compile(`([A-Za-z]+) $`)
	capitalizedBeforePattern = compile(`\b([A-Z][a-z]+) $`)
	lowerWordAfterPattern    = compile(`^ ([a-z]+)\b`)
	numberAfterPattern       = compile(`^[.,:]\d`)
)

// contextBytes bounds the text a rule looks at around its match, so a rule on a long text stays linear.
const contextBytes = 200

func previousCharacter(m match) rune {
	if m.start() == 0 {
		return ' '
	}
	r, _ := utf8.DecodeLastRuneInString(m.text[:m.start()])
	return r
}

func nextCharacter(m match) rune {
	if m.end() == len(m.text) {
		return ' '
	}
	r, _ := utf8.DecodeRuneInString(m.text[m.end():])
	return r
}

// spot is a match and the text around it, as the rules that read context see it.
type spot struct {
	match
	preceding, following string
}

func spotOf(m match) spot {
	return spot{m, textBefore(m), textAfter(m)}
}

func textBefore(m match) string {
	return before(m.text, m.start(), contextBytes)
}

func textAfter(m match) string {
	return after(m.text, m.end(), contextBytes)
}

// nextWord is the kind of the first word after end, past closing quotes and brackets and one run of spaces.
func nextWord(text string, end int) wordKind {
	following := after(text, end, contextBytes)
	closing, _ := closingPattern.find(following)
	following = following[closing.end():]
	if strings.TrimSpace(following) == "" {
		return wordNone
	}

	first, _ := utf8.DecodeRuneInString(following)
	if !unicode.IsSpace(first) {
		return wordOther
	}

	following = strings.TrimLeftFunc(following, unicode.IsSpace)
	first, _ = utf8.DecodeRuneInString(following)
	switch {
	case first == '(' || first == '[' || unicode.IsDigit(first):
		return wordOther
	case quotedPattern.MatchString(following):
		return wordQuoted
	case initialPattern.MatchString(following):
		return wordInitial
	}

	found, ok := leadingLettersPattern.find(following)
	if !ok {
		return wordOther
	}

	word := found.group(0)
	switch {
	case sentenceStarters[word]:
		return wordStarter
	case len(word) == 1 && isUpper(word):
		return wordLetter
	case isUpper(word):
		return wordCapitals
	case isUpper(word[:1]):
		return wordName
	}
	return wordOther
}

// sentenceEnd is whether a period just before end also ends its sentence, so a rule that reads the word before it
// keeps it.
func sentenceEnd(text string, end int) bool {
	if strings.HasPrefix(text[end:], "..") {
		return true
	}

	// A bracket opens a sentence only before a sentence starter ("(The") or a stage direction in capitals ("(FREDERIC
	// enters)").
	if bracketed, ok := openingBracketPattern.find(text[end:]); ok {
		word := bracketed.group(1)
		return sentenceStarters[word] || (len(word) > 1 && isUpper(word))
	}
	return sentenceStartKinds[nextWord(text, end)]
}

func continuesBefore(m match) bool {
	previous := previousCharacter(m)
	return isAlphanumeric(previous) || strings.ContainsRune("./-#:+_", previous)
}

func continuesAfter(m match) bool {
	following := nextCharacter(m)
	return isAlphanumeric(following) || strings.ContainsRune("-/_", following) ||
		numberAfterPattern.MatchString(textAfter(m))
}

// continuesNumber is whether the match is part of a longer number, code or address on either side.
func continuesNumber(m match) bool {
	return continuesBefore(m) || continuesAfter(m)
}

// before is up to count bytes of text before offset, widened to the start of the character there.
func before(text string, offset, count int) string {
	start := max(0, offset-count)
	for start > 0 && !utf8.RuneStart(text[start]) {
		start--
	}
	return text[start:offset]
}

// after is up to count bytes of text from offset, widened to the end of the character there.
func after(text string, offset, count int) string {
	end := min(len(text), offset+count)
	for end < len(text) && !utf8.RuneStart(text[end]) {
		end++
	}
	return text[offset:end]
}

func sentencePeriod(text string, end int) string {
	if sentenceEnd(text, end) {
		return "."
	}
	return ""
}

// keptPeriod is the period mark a match wrote, kept only where it also ends the sentence.
func keptPeriod(m match, mark string) string {
	if mark == "" {
		return ""
	}
	return sentencePeriod(m.text, m.end())
}
