package text

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

var (
	greekNames = strings.Fields("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron " +
		"pi rho sigma tau upsilon phi chi psi omega")
	// Greek letters and roots named before the number rules read their operands ("\u221a16" as "square root of 16",
	// "3\u03c0"); the ohm sign stays for the unit rules ("10 k\u03a9") and is named at the end.
	earlySymbols       = earlySymbolTable()
	earlySymbolPattern = compile(`[\x{3b1}-\x{3c9}\x{391}-\x{3a1}\x{3a3}-\x{3a8}\x{221a}\x{221b}\x{221e}` +
		`\x{2669}\x{266a}]`)
	// A slash between two Greek-letter terms is a ratio ("\u0394y/\u0394x" as "delta y over delta x").
	greekRatioPattern = compile(`\b((?:` + strings.Join(greekNames, "|") + `) ?[a-z]?)/((?:` +
		strings.Join(greekNames, "|") + `) ?[a-z]?)\b`)
	// A slash between a letter and a letter or number is a ratio ("a/b", "x/2"), except in shorthand ("w/o", "n/a").
	variableRatioPattern = compile(`\b([a-z])/([a-z]|\d{1,3})\b`)
	slashWords           = map[string]string{"w/o": "without", "c/o": "care of", "b/c": "because"}
	slashShorthand       = set("n/a", "w/e", "a/c", "i/o", "y/n", "p/e", "a/v", "r/c")
	// A tilde before a number approximates it ("~5 km"); joined to a word it is part of a name ("HEAD~3") and elsewhere
	// a path ("~/projects").
	tildeNumberPattern = compile(`(^|[^A-Za-z0-9])~ ?(\d)`)
	// "#" after a note letter is "sharp" ("C#", "F# minor"), not "hash".
	sharpPattern = compile(`\b([A-G])(#|\x{266f})`)
	// Operators read between their operands, before the number rules.
	symbolForms = newSymbolWords(map[string]string{"\u00b1": "plus or minus", "\u00d7": "times", "\u00f7": "divided by",
		"\u2212": "minus", "=": "equals", "\u2248": "approximately equals", "\u2260": "is not equal to"})
	// Symbols left after every other rule, read as words.
	leftoverSymbols = newSymbolWords(leftoverSymbolTable())
	// Marks with no spoken form that are punctuation rather than symbols: bullets, daggers and reference marks.
	decorativeMarks = "\u2022\u2023\u2043\u2020\u2021\u203b\u25cf\u25cb\u25a0\u25a1"
)

// symbolWords reads each symbol of a table as its words, set apart by spaces.
type symbolWords struct {
	pattern *pattern
	words   map[string]string
}

func newSymbolWords(words map[string]string) symbolWords {
	return symbolWords{compile(`\s{0,3}(` + longestFirst(sortedKeys(words)) + `)\s{0,3}`), words}
}

func (s symbolWords) read(text string) string {
	return s.pattern.replaceFunc(text, func(m match) string { return " " + s.words[m.group(1)] + " " })
}

// earlySymbolTable names the math, music and Greek letter symbols the early symbol rule reads.
func earlySymbolTable() map[rune]string {
	names := map[rune]string{
		'\u221a': "square root of", '\u221b': "cube root of", '\u221e': "infinity", '\u2669': "quarter note",
		'\u266a': "eighth note", '\u03c2': "sigma",
	}
	for i, name := range greekNames {
		// The Greek block has a final sigma after rho and no capital there, so later letters sit one code point
		// further.
		skip := 0
		if i >= 17 {
			skip = 1
		}
		names[rune(0x3b1+i+skip)] = name
		if upper := rune(0x391 + i + skip); upper != '\u03a9' {
			names[upper] = name
		}
	}
	return names
}

// earlySymbol is a symbol's name, set apart by spaces from the letters and digits it touches.
func earlySymbol(m match) string {
	symbol, _ := utf8.DecodeRuneInString(m.group(0))
	name := earlySymbols[symbol]
	if m.start() > 0 && isAlphanumeric(previousCharacter(m)) {
		name = " " + name
	}

	if m.end() < len(m.text) && isAlphanumeric(nextCharacter(m)) {
		name += " "
	}
	return name
}

// earlySymbolWords names Greek letters, roots and note signs, reads a slash between Greek terms as "over", and a tilde
// before a number as "about".
func earlySymbolWords(text string) string {
	text = tildeNumberPattern.ReplaceAllString(text, "${1}about $2")
	text = earlySymbolPattern.replaceFunc(text, earlySymbol)
	return greekRatioPattern.ReplaceAllString(text, "$1 over $2")
}

// variableRatio reads "a/b" as "a over b" and "w/o" as "without", and declines for shorthand said as written ("n/a").
func variableRatio(m match) (string, bool) {
	written := m.group(0)
	if words, ok := slashWords[written]; ok {
		return words, true
	}

	if slashShorthand[written] {
		return "", false
	}
	return m.group(1) + " over " + m.group(2), true
}

// leftoverSymbolWords reads a slash between letters as "over", names every symbol left, and replaces one with no
// spoken form ("\u2022", "\u2020", "\u2605") by a space, since a reader says nothing for it.
func leftoverSymbolWords(text string) string {
	text = variableRatioPattern.replaceWhere(text, variableRatio)
	text = leftoverSymbols.read(text)
	return strings.Map(func(r rune) rune {
		if unicode.Is(unicode.S, r) || strings.ContainsRune(decorativeMarks, r) {
			return ' '
		}
		return r
	}, text)
}

// leftoverSymbolTable names each symbol left after every other rule, the money signs by their currency.
func leftoverSymbolTable() map[string]string {
	words := map[string]string{
		"&": "and", "+": "plus", "@": "at", "#": "hash", "/": "slash", "%": "percent", "\u2030": "per mille",
		"\u00b0": degreeUnit.plural, "\u2103": degreeUnit.plural + " Celsius",
		"\u2109": degreeUnit.plural + " Fahrenheit",
		"\u00a2": currencies["$"].minor.plural, "\u00b6": "paragraph", "\u00a7": "section", "\u00b5": "mu",
		"\u03a9": "omega", "^": "to the power of", "|": "or", "\u2208": "in", "\u2116": "number", "\u266d": "flat",
		"\u00a9": "copyright", "\u00ae": "registered", "\u2122": "trademark", "\u2192": "to", "~": "tilde",
		"\u2211": "sum", "\u222b": "integral", "\u2202": "partial", "\u2032": "prime", "\u2033": "double prime",
	}
	for _, sign := range moneySigns {
		words[string(sign)] = currencies[string(sign)].major.plural
	}

	// The bitcoin sign alone names the currency.
	words["\u20bf"] = currencies["\u20bf"].major.singular
	for _, comparison := range []string{"<", ">", "\u2264", "\u2265"} {
		words[comparison] = comparisonWords[comparison]
	}
	return words
}
