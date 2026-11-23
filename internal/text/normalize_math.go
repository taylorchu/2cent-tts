package text

var (
	variableComparisonPattern = compile(`\b([A-Za-z]) ?(<=|>=|\x{2264}|\x{2265}|<|>) ?`)
	spacedMinusPattern        = compile(`(\d) - (\d)`)
	spacedDividePattern       = compile(`(\d) / (\d)`)
	dividePattern             = compile(`\b([Dd]ivide \S+) ?\x{f7} ?`)
	comparisonWords           = comparisonTable()
	powerPattern              = compile(`\b([A-Za-z0-9]{1,30})\^(\d{1,3})\b`)
	starProductPattern        = compile(`(\d) ?\* ?(\d)`)
	// A multiplier standing alone ("3x zoom") says the letter, and "times" before a comparison ("3x faster"); inside
	// a name it stays ("logo@2x.png").
	multiplierPattern      = compile(`(^|[\s(])(\d+(?:\.\d+)?)x\b`)
	comparisonAfterPattern = compile(`^ (?:as|more|less|fewer|the|[a-z]+er)\b`)
)

// multiplier reads "3x faster" as "3 times faster" and "3x zoom" as "3 X zoom".
func multiplier(m match) string {
	if comparisonAfterPattern.MatchString(textAfter(m)) {
		return m.group(1) + m.group(2) + " times"
	}
	return m.group(1) + m.group(2) + " X"
}

// power reads "x^2" as "x squared" and "2^10" as "2 to the tenth".
func power(m match) string {
	return powerWords(m.group(1), m.group(2))
}

// variableComparison reads "n > 5" as "n is greater than 5".
func variableComparison(m match) string {
	return m.group(1) + " is " + comparisonWords[m.group(2)] + " "
}

var (
	byPattern         = compile(`\b(\d)x(\d)\b`)
	spacedByPattern   = compile(`(\d(?: ?[A-Za-z]{1,2})?) x (\d)`)
	dimensionsPattern = compile(`(\d(?: ?[A-Za-z]{1,2})?) ?\x{d7} ?(\d)`)
	sumNearPattern    = compile(`[=+]|\s-\s`)
)

// dimensions reads "3 \u00d7 4" as "3 by 4", but "times" in a sum ("3 + 4 \u00d7 2").
func dimensions(m match) (string, bool) {
	if sumNearPattern.MatchString(textBefore(m) + m.group(0) + textAfter(m)) {
		return "", false
	}
	return m.group(1) + " by " + m.group(2), true
}

// powerWords reads base raised to exponent: "x squared", "x cubed", "2 to the tenth".
func powerWords(base, exponent string) string {
	switch exponent {
	case "2":
		return base + " squared"
	case "3":
		return base + " cubed"
	}
	return base + " to the " + ordinalWords(integerTextWords(exponent))
}

func comparisonTable() map[string]string {
	words := map[string]string{"<": "less than", ">": "greater than", "<=": "less than or equal to",
		">=": "greater than or equal to"}
	words["\u2264"], words["\u2265"] = words["<="], words[">="]
	return words
}
