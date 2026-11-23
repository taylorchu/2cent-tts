package text

import (
	"regexp"
	"strings"
)

// pattern is a rule's regular expression with the readers that hand the rules a match.
type pattern struct {
	*regexp.Regexp
}

func compile(expression string) *pattern {
	return &pattern{regexp.MustCompile(expression)}
}

// repeated writes group{least,most} as nested optional groups, the same match order, since RE2 refuses a nested
// repeat whose counts multiply past a thousand.
func repeated(group string, least, most int) string {
	optional := ""
	for range most - least {
		optional = "(?:" + group + optional + ")?"
	}
	return strings.Repeat(group, least) + optional
}

// match is one match of a pattern against text, its offsets in bytes.
type match struct {
	text  string
	index []int
}

func (m match) start() int {
	return m.index[0]
}

func (m match) end() int {
	return m.index[1]
}

func (m match) group(i int) string {
	if m.index[2*i] < 0 {
		return ""
	}
	return m.text[m.index[2*i]:m.index[2*i+1]]
}

func (p *pattern) find(text string) (match, bool) {
	index := p.FindStringSubmatchIndex(text)
	return match{text, index}, index != nil
}

// firstGroup is the first group of the first match in text, or "" where none matches.
func (p *pattern) firstGroup(text string) string {
	if m, ok := p.find(text); ok {
		return m.group(1)
	}
	return ""
}

func (p *pattern) findAll(text string) []match {
	var matches []match
	for _, index := range p.FindAllStringSubmatchIndex(text, -1) {
		matches = append(matches, match{text, index})
	}
	return matches
}

// replaceFunc writes speak's reading in place of every match, each read against the whole original text.
func (p *pattern) replaceFunc(text string, speak func(match) string) string {
	return p.replaceWhere(text, func(m match) (string, bool) { return speak(m), true })
}

// replaceWhere is replaceFunc for a speak that may decline a match, which is then kept as written.
func (p *pattern) replaceWhere(text string, speak func(match) (string, bool)) string {
	indexes := p.FindAllStringSubmatchIndex(text, -1)
	if indexes == nil {
		return text
	}

	var b strings.Builder
	last := 0
	for _, index := range indexes {
		spoken, ok := speak(match{text, index})
		if !ok {
			continue
		}

		b.WriteString(text[last:index[0]])
		b.WriteString(spoken)
		last = index[1]
	}
	b.WriteString(text[last:])
	return b.String()
}

// distinctGroups is how many different first groups the matches in text hold.
func (p *pattern) distinctGroups(text string) int {
	seen := map[string]bool{}
	for _, found := range p.findAll(text) {
		seen[found.group(1)] = true
	}
	return len(seen)
}
