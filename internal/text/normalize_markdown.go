package text

import (
	"strconv"
	"strings"
	"unicode"
)

const emphasized = `([^\s*_](?:[^\n*]*?[^\s*_])?)`

var (
	codeFencePattern = compile(`(?m)^[ \t]*(?:\x60\x60\x60|~~~)[^\n]*$`)
	headingPattern   = compile(`(?m)^[ \t]{0,3}#{1,6}[ \t]+([^\n]*?)[ \t#]*$`)
	bulletPattern    = compile(`(?m)^[ \t]*[-*+][ \t]+(?:\[[ xX]\][ \t]+)?([^\n]*)$`)
	numberedPattern  = compile(`(?m)^[ \t]*(\d{1,3})[.)][ \t]+([^\n]*)$`)
	// A ">" opening a line is a quote mark before a space and a list number or no digit, or before the line's end,
	// since ">5 years" and "> 5 years" compare.
	quotePattern = compile(`(?m)^[ \t]*>(?:[ \t]*>)*(?:[ \t]+([^\n0-9 \t][^\n]*|\d{1,3}[.)][ \t][^\n]*))?[ \t]*$`)
	// A number item of a list written in one line ("Steps: 1. Open Settings 2. Tap Wi-Fi").
	inlineItemPattern = compile(`(^|[ \t])(\d{1,3})[.)][ \t]+`)
	// Text before an inline list's first item: the line's start, a colon or a sentence end.
	listOpeningPattern = compile(`(?:^|[:.!?])[ \t]*$`)
	rulePattern        = compile(`(?m)^[ \t]*(?:[-*_][ \t]*){3,}$`)
	linkPattern        = compile(`!?\[([^\]\n]*)\]\([^)\s]*\)`)
	// Emphasis marks are one to three stars, or two or three underscores, around text that starts and ends on no space.
	emphasisPattern = compile(`\*\*\*` + emphasized + `\*\*\*|\*\*` + emphasized + `\*\*|\*` + emphasized + `\*|___` +
		emphasized + `___|__` + emphasized + `__`)
)

// markdownSentence ends line as a sentence, so the next line does not run on from it.
func markdownSentence(line string) string {
	line = strings.TrimRightFunc(line, unicode.IsSpace)
	if line == "" || strings.ContainsRune(".!?:;", rune(line[len(line)-1])) {
		return line
	}
	return line + "."
}

// inlineList reads a list written in line as its own lines are: "Steps: 1. Open 2. Tap" as "Steps: 1, Open. 2, Tap".
// A list counts up from 1 after the line's start, a colon or a sentence end, so "I have 2. Then" stays.
func inlineList(line string) string {
	var items []match
	for _, m := range inlineItemPattern.findAll(line) {
		number, _ := strconv.Atoi(m.group(2))
		if number == 1 && listOpeningPattern.MatchString(line[:m.index[4]]) {
			items = []match{m}
		} else if items != nil && number == len(items)+1 {
			items = append(items, m)
		}
	}

	if len(items) < 2 {
		return line
	}

	var b strings.Builder
	last := 0
	for k, m := range items {
		preceding := line[last:m.index[4]]
		if k > 0 {
			preceding = strings.TrimRight(preceding, " \t")
			if preceding != "" && !strings.ContainsRune(".!?:;,", rune(preceding[len(preceding)-1])) {
				preceding += "."
			}
			preceding += " "
		}
		b.WriteString(preceding + m.group(2) + ", ")
		last = m.end()
	}
	b.WriteString(line[last:])
	return markdownSentence(b.String())
}

// quoteOpens is whether text opens with a quote mark; the first match starts where any match can.
func quoteOpens(text string) bool {
	m, ok := quotePattern.find(text)
	return ok && m.start() == 0
}

// markdownPlain drops Markdown marks: "## Setup" as "Setup.", "- Fast setup" as "Fast setup.", "1. Open **Settings**"
// as "1, Open Settings.", "> quoted" as "quoted", "[docs](https://x.io)" as "docs"; a list item keeps its number, read
// with a comma, since a number alone as a sentence ("one.") is no form training text holds.
func markdownPlain(text string) string {
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = inlineList(line)
	}
	text = strings.Join(lines, "\n")
	if !strings.Contains(text, "\n") && !linkPattern.MatchString(text) && !quoteOpens(text) {
		return text
	}

	text = quotePattern.replaceFunc(text, func(m match) string { return markdownSentence(m.group(1)) })
	text = rulePattern.ReplaceAllString(codeFencePattern.ReplaceAllString(text, ""), "")
	text = linkPattern.ReplaceAllString(text, "$1")
	text = headingPattern.replaceFunc(text, func(m match) string { return markdownSentence(m.group(1)) })
	text = bulletPattern.replaceFunc(text, func(m match) string { return markdownSentence(m.group(1)) })
	text = numberedPattern.replaceFunc(text, func(m match) string {
		return m.group(1) + ", " + markdownSentence(m.group(2))
	})
	return withoutEmphasis(text)
}

// withoutEmphasis keeps the text inside emphasis marks.
func withoutEmphasis(text string) string {
	return emphasisPattern.ReplaceAllString(text, "$1$2$3$4$5")
}
