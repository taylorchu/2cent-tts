package text

import "strings"

// Chat shorthand reads as the words it stands for ("btw" as "by the way", "u" as "you"), only in lowercase or
// capitalized, since in capitals several are other things ("RN" a nurse). Book text spells letters ("the u s a") and
// scans break words ("crept np"), so "u", "ur" and "np" need chat-shaped words around them.
var (
	chatWordNames = map[string]string{
		"omw": "on my way", "btw": "by the way", "tmrw": "tomorrow", "bday": "birthday",
		"ttyl": "talk to you later", "ngl": "not gonna lie", "idk": "I don't know", "idc": "I don't care",
		"imo": "in my opinion", "imho": "in my humble opinion", "tbh": "to be honest", "brb": "be right back",
		"lmk": "let me know", "smh": "shaking my head", "rn": "right now", "pls": "please", "plz": "please",
		"thx": "thanks", "nvm": "never mind", "jk": "just kidding", "ikr": "I know right", "irl": "in real life",
		"afaik": "as far as I know", "wyd": "what are you doing", "hbu": "how about you", "wbu": "what about you",
		"b4": "before", "gr8": "great", "ppl": "people", "abt": "about",
	}
	chatForms   = chatFormTable()
	chatPattern = compile(`\b(` + longestFirst(sortedKeys(chatForms)) + `)\b`)
	tldrPattern = compile(`\b(?:TL;DR|tl;dr|TLDR|tldr)\b`)
	// "u" and "ur" with a whole lowercase word or the text's edge on each side ("can u grab", "see u tmrw").
	youPattern = compile(`(^|\b[A-Za-z0-9']+,? )(u|ur)( [a-z']+\b|[,!?]|$)`)
	// Words before a letter named as a letter ("the silent u").
	letterBefore = set("the", "a", "an", "silent", "letter", "capital", "small", "vowel", "final", "long", "short")
	// A letter right after "u," ("u,s.").
	letterAfterPattern = compile(`^[A-Za-z]\b`)
	// "np" standing alone as a reply ("np, ttyl!", "thanks! np").
	noProblemPattern    = compile(`(^|[.!?,] )([Nn])p([.!?,]|$)`)
	withPattern         = compile(`\b([Ww])/ `)
	chatJoinedPattern   = compile(`^(?:[.'\-][A-Za-z]|['"])`)
	sentenceStopPattern = compile(`[.!?]\s*$`)
	// A smiley set apart from words (":)", ";-)", ":D", ":P") is not said; "Note:)" has no space before it.
	emoticonPattern      = compile(`(^|[ \t\r\n])[:;]-?(?:\)+|\(+|[DPp])`)
	sentenceAfterPattern = compile(`^[ \t\r\n]+[A-Z]`)
)

// chatFormTable maps each chat shorthand, lowercase or capitalized, and each growth period ("YoY") to its words.
func chatFormTable() map[string]string {
	forms := map[string]string{"YoY": "year over year", "MoM": "month over month", "QoQ": "quarter over quarter"}
	for word, words := range chatWordNames {
		forms[word] = words
		forms[capitalize(word)] = words
	}
	return forms
}

// chatWord reads a chat shorthand as its words, or declines where it is quoted or joined to more text ("'pls' package",
// "mod'rn"), or capitalized away from a sentence start ("Imo State" is a place).
func chatWord(m match) (string, bool) {
	written, words := m.group(1), chatForms[m.group(1)]
	if !strings.ContainsRune(" (", previousCharacter(m)) || chatJoinedPattern.MatchString(textAfter(m)) {
		return "", false
	}

	preceding := textBefore(m)
	if isTitle(written) && strings.TrimSpace(preceding) != "" && !sentenceStopPattern.MatchString(preceding) {
		return "", false
	}

	if isTitle(written) {
		return strings.ToUpper(words[:1]) + words[1:], true
	}
	return words, true
}

// you reads "u" as "you" and "ur" as "your", or declines beside a lone letter or a word that names a letter, a letter
// right after its comma among them ("u,s."); a number before it needs a comma between ("at 8, ur place").
func you(m match) (string, bool) {
	preceding, word, following := m.group(1), m.group(2), m.group(3)
	previous := strings.Trim(preceding, " ,")
	if isDigits(previous) {
		previous = "0"
		if strings.HasSuffix(preceding, ", ") {
			previous = ""
		}
	}

	rest := strings.TrimSpace(following)
	_, letterAfter := letterAfterPattern.find(textAfter(m))
	if (previous != "" && (len(previous) < 2 || !isLower(previous) || letterBefore[previous])) ||
		(len(rest) == 1 && isAlpha(rest)) || (!isAlpha(rest) && letterAfter) {
		return "", false
	}

	if word == "u" {
		return preceding + "you" + following, true
	}
	return preceding + "your" + following, true
}

// emoticon drops a smiley, keeping a line break before it and a bracket it closes ("(great :))"), or declines where
// a word goes on from it (":Dear"); a smiley between a word and a capital ends the sentence ("Thanks :) See").
func emoticon(m match) (string, bool) {
	if !strings.ContainsRune(" \t\r\n.,!?", nextCharacter(m)) {
		return "", false
	}

	kept := m.group(1)
	if kept == " " || kept == "\t" {
		kept = ""
	}

	closing := ""
	preceding := textBefore(m)
	if strings.HasSuffix(m.group(0), "))") && strings.Count(preceding, "(") > strings.Count(preceding, ")") {
		closing = ")"
	}

	preceding = strings.TrimRight(preceding, " \t\r\n")
	if preceding != "" && !strings.ContainsRune(".!?,;:", rune(preceding[len(preceding)-1])) &&
		sentenceAfterPattern.MatchString(textAfter(m)) {
		return closing + ".", true
	}
	return kept + closing, true
}

// chatWords reads chat shorthand as words.
func chatWords(text string) string {
	text = emoticonPattern.replaceWhere(text, emoticon)
	text = tldrPattern.ReplaceAllString(text, "T L D R")
	text = withPattern.ReplaceAllString(text, "${1}ith ")
	text = noProblemPattern.ReplaceAllString(text, "${1}${2}o problem$3")
	text = chatPattern.replaceWhere(text, chatWord)
	return youPattern.replaceWhere(text, you)
}
