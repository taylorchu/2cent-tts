package text

import (
	"slices"
	"strings"
)

var (
	// A version after "v", a version word, or a letter prefix joined by a hyphen or dot ("FW-1.2.3", the lineage
	// "B.1.1.7").
	versionPattern = compile(`\b(?:([Vv])|((?i:version|release|patch|firmware|kernel|python|node|java|ios|macos` +
		`|android|ubuntu)) |([A-Z]{1,4})[-.])?(\d+(?:\.\d+)+)(-\d+)?\b`)
	parenLetterPattern = compile(`\b(\d{3})\(([a-z])\)`)
	teamNumberPattern  = compile(`\b(\d{2})ers\b`)
	aircraftPattern    = compile(`\b(Boeing|Airbus|Cessna|Embraer|Lockheed|Douglas) (A?)(\d{3})\b`)
	// A flight number after "flight" and its airline code ("Flight UA 837"), read in pairs as a flight number is
	// said.
	flightPattern = compile(`\b((?i:flights?) )([A-Z][A-Z0-9]|[0-9][A-Z]) ?(\d{1,4})\b`)
	// An HTTP status code next to a word that marks it ("returns a 404", "500 error"), said as developers say it
	// ("four oh four"); "returns 450 units" after a marking word alone stays a count.
	statusCodePattern = compile(`\b(?:((?i:HTTP|status(?: code)?|error(?: code)?|returns?|returned|returning` +
		`|throws?|threw|throwing))( an?)? )?([45]\d\d)\b( (?i:errors?|page|response|status|not found|forbidden` +
		`|unauthorized|bad request|bad gateway|internal server error|service unavailable|too many requests)\b)?`)
	statusNextPattern = compile(
		`^(?:[.,;:!?)]|$| (?:when|if|for|on|because|and|or|but|instead|from|with|in)\b)`)
	// A post's place in a thread ("(1/7)") and a part of a whole after a label ("Part 2/5") are "one of seven".
	threadNumberPattern = compile(`\((\d{1,2})/(\d{1,2})\)`)
	labeledPartPattern  = compile(`\b((?i:part|step|episode|chapter|lesson|slide|page|day|week|round|game|question` +
		`|stage|level|season|issue|disc|track) )(\d{1,3})/(\d{1,3})\b`)
	sectionSignPattern = compile(`\x{a7}(\x{a7}?) ?`)
	// Degrees, minutes and seconds of a coordinate ("40\u00b042'46\"N").
	coordinatePattern = compile(
		`(\d{1,3})\x{b0} ?(\d{1,2})['\x{2032}] ?(?:(\d{1,2}(?:\.\d{1,6})?)["\x{2033}] ?)?([NSEW]\b)?`)
	// An upgrade or release within three words before two dotted numbers makes them a version ("upgrade from 2.16
	// to").
	versionContextPattern = compile(`\b(?i:version|v|build|release) $|\b(?i:upgrad\w*|updat\w*) from (?:\S+ to )?$`)
	keyNames              = map[string]string{"Ctrl": "control", "Control": "control", "Alt": "alt", "Shift": "shift",
		"Cmd": "command", "Command": "command", "Opt": "option", "Option": "option", "Fn": "function",
		"Win": "windows"}
	// A key chord ("Ctrl+Shift+T") names each key, without "plus".
	keyChordPattern = compile(`\b((?:(?:` + longestFirst(sortedKeys(keyNames)) + `)\+)+)([A-Za-z0-9]+)\b`)
	sizeNames       = map[string]string{"XXS": "double extra small", "XS": "extra small", "S": "small",
		"M": "medium", "L": "large", "XL": "extra large", "XXL": "double extra large", "XXXL": "triple extra large"}
	sizeRangePattern = compile(`\b(X{0,3}S|M|X{0,3}L)-(X{0,3}S|M|X{0,3}L)\b`)

	versionAfterPattern = compile(`^\.\d`)
)

// versionPart reads one part of a version: up to two digits a whole number, "04" as "oh four", three or four digits
// in pairs ("551" as "five fifty one"), longer parts and the parts of an IP address digit by digit.
func versionPart(part string, address bool) string {
	switch {
	case address || len(part) >= 5:
		return digitWords(part)
	case len(part) >= 3:
		return houseNumberWords(part)
	case len(part) == 2 && part[0] == '0':
		return twoDigitWords(part)
	}
	return part
}

// version reads "v2.4.1" as "version 2 point 4 point 1" and "551.86.10" as "five fifty one point 86 point 10";
// two parts are a version only after "v", a version word, or an upgrade or release nearby.
func version(m match) (string, bool) {
	letter, word, prefix, dotted, suffix := m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
	parts := strings.Split(dotted, ".")
	previous := previousCharacter(m)
	if previous == '.' || previous == '-' || versionAfterPattern.MatchString(textAfter(m)) {
		return "", false
	}

	labeled := letter != "" || word != "" || versionContextPattern.MatchString(textBefore(m))
	if (!labeled && len(parts) < 3) || (prefix != "" && isAlphanumeric(previous)) {
		return "", false
	}

	address := letter == "" && word == "" && addressParts(parts)
	spoken := spokenVersion(parts, address, suffix)
	switch {
	case letter != "":
		return "version " + spoken, true
	case prefix != "":
		return spelled(prefix) + " " + spoken, true
	case word != "":
		return word + " " + spoken, true
	}
	return spoken, true
}

// spokenVersion reads a version's parts with "point" between them, or an IP address's with "dot", and its suffix.
func spokenVersion(parts []string, address bool, suffix string) string {
	joiner := " point "
	if address {
		joiner = " dot "
	}

	spoken := joined(parts, joiner, func(part string) string { return versionPart(part, address) })
	if suffix != "" {
		spoken += " dash " + suffix[1:]
	}
	return spoken
}

func sectionSign(m match) string {
	if m.group(1) != "" {
		return "sections "
	}
	return "section "
}

// coordinate reads "40\u00b042'46\"N" as "40 degrees 42 minutes 46 seconds north".
func coordinate(m match) string {
	degreeCount, minutes, seconds, point := m.group(1), m.group(2), m.group(3), m.group(4)
	words := degreeCount + " " + degreeUnit.counted(degreeCount) + " " + minutes + " " +
		minuteUnit.counted(minutes)
	if seconds != "" {
		words += " " + seconds + " " + secondUnit.counted(seconds)
	}

	if point != "" {
		words += " " + compass[point]
	}
	return words
}

// addressParts is whether a version's parts could be an IP address's: four numbers up to 255.
func addressParts(parts []string) bool {
	return len(parts) == 4 && !slices.ContainsFunc(parts, func(part string) bool { return number(part) > 255 })
}

// teamNumber reads "49ers" as "forty niners".
func teamNumber(m match) string {
	words := integerWords(number(m.group(1)))
	if strings.HasSuffix(words, "e") {
		return words + "rs"
	}
	return words + "ers"
}

// aircraft reads "Boeing 747" as "Boeing seven forty seven".
func aircraft(m match) string {
	maker, letter := m.group(1), m.group(2)
	if letter != "" {
		letter += " "
	}
	return maker + " " + letter + houseNumberWords(m.group(3))
}

// flight reads "Flight UA 837" as "Flight UA eight thirty seven".
func flight(m match) string {
	return m.group(1) + m.group(2) + " " + houseNumberWords(m.group(3))
}

// statusCode reads "returns a 404" as "returns a four oh four" and "500 error" as "five hundred error".
func statusCode(m match) (string, bool) {
	marking, articleWord, code, following := m.group(1), m.group(2), m.group(3), m.group(4)
	if continuesNumber(m) || (following == "" && (marking == "" || !statusNextPattern.MatchString(textAfter(m)))) {
		return "", false
	}

	lead := ""
	if marking != "" {
		lead = marking + articleWord + " "
	}
	return lead + houseNumberWords(code) + following, true
}

// partOf reads part "2" of whole "5" as "2 of 5", or declines for a part larger than its whole.
func partOf(part, whole string) (string, bool) {
	if 0 < number(part) && number(part) <= number(whole) && number(whole) >= 2 {
		return part + " of " + whole, true
	}
	return "", false
}

// threadNumber reads "(1/7)" at either end of a post as "1 of 7".
func threadNumber(m match) (string, bool) {
	if strings.TrimSpace(textBefore(m)) != "" && strings.Trim(textAfter(m), " .!?") != "" {
		return "", false
	}
	return partOf(m.group(1), m.group(2))
}

// labeledPart reads "Part 2/5" as "Part 2 of 5".
func labeledPart(m match) (string, bool) {
	spoken, ok := partOf(m.group(2), m.group(3))
	if !ok {
		return "", false
	}
	return m.group(1) + spoken, true
}

// keyChord reads "Ctrl+Shift+T" as "control shift T".
func keyChord(m match) string {
	keys := strings.Split(strings.TrimRight(m.group(1), "+"), "+")
	return joined(keys, " ", func(key string) string { return keyNames[key] }) + " " + m.group(2)
}

// sizeRange reads "M-XXL" as "medium to double extra large"; a pair of one-letter sizes ("S-M class") is too often
// something else.
func sizeRange(m match) (string, bool) {
	start, end := m.group(1), m.group(2)
	if max(len(start), len(end)) < 2 {
		return "", false
	}
	return sizeNames[start] + " to " + sizeNames[end], true
}
