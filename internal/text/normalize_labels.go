package text

import (
	"slices"
	"strings"
)

// strongLabels are labels before a number that always make it an identifier ("Docket No. 2021-3456", "Public
// Law 104-191").
var strongLabels = set("docket", "law", "statute", "patent", "serial", "sn", "pn", "isbn", "vin", "sku", "tracking",
	"confirmation", "booking")

// legalLabels are labels of legal numbers, whose first group reads as a year where it is one ("Docket No. 2021-3456").
var legalLabels = set("docket", "law", "statute", "patent")

// referenceLabels mark a number as a reference, so a date after one is no date ("Order 2026-12-01"), but alone they
// make no identifier ("In any case 300-400 men"); "No." or "#" does ("Part No. 330-12").
var referenceLabels = set("no", "number", "order", "invoice", "account", "ticket", "id", "pin", "code", "payment",
	"part", "item", "case", "claim", "policy", "reference", "ref")

// phoneLabels are the words after which a phone-shaped pair is a phone number ("Call 555-0199", "call us on
// 555-0199").
var phoneLabels = set("call", "dial", "phone", "tel", "fax", "ring", "text", "mobile", "cell", "hotline", "helpline",
	"reach", "contact", "telephone")

var (
	identifierWordsSet = union(strongLabels, referenceLabels, phoneLabels)
	// A reference or phone label phrase right before a number: the label, its "number" or "code", "is", one code
	// word in capitals and a preposition at most ("Order No. 7741890", "Text STOP to 88800").
	referenceBeforePattern = compile(`\b(?i:` + strings.Join(sortedKeys(union(referenceLabels, phoneLabels)), "|") +
		`)\b[.:]?(?: (?i:number|no\.?|code|id)|#)?(?: (?i:is|was))?:?(?: [A-Z0-9]{2,})?(?: (?i:to|at|on))? #?$`)
	phoneBeforePattern = compile(`\b(?i:` + strings.Join(sortedKeys(phoneLabels), "|") +
		`)(?: (?:us|me|him|her|them))?(?: (?:at|on))?:? $`)
	// After these a bare number labels one thing and is read in pairs ("room four twelve"); before them it counts.
	pairedNouns = set("room", "suite", "flight", "gate", "locker", "platform", "berth", "seat", "apartment", "bus",
		"route", "unit", "cabin", "pier", "terminal", "model", "highway", "channel", "rtx", "gtx", "rx", "radeon",
		"geforce", "ryzen", "xeon", "series", "iphone", "galaxy", "pixel", "thinkpad")
	pairedNumberPattern = compile(`\b((?i:` + strings.Join(sortedKeys(pairedNouns), "|") +
		`)s? (?:#|(?i:no\.?|number) )?)(\d{3,4})(?: to (\d{3,4}))?\b`)
	// Words around a hyphened pair that make it a cricket total ("250-7" as "two hundred and fifty for seven").
	cricketPattern = compile(`\b(?i:runs?|wickets?|innings|overs?|declared|all out|cricket|batting|bowled|bowling)\b`)
	// The word right before a number, or the label before "No." or "#" ("Docket No.", "Case #").
	labelPattern = compile(`(?:^|[^A-Za-z/])([A-Za-z/]+)\.?:? ((?:[Nn]o|[Nn]umber)\.? |#)?$`)
	// Label words after which a letter is a label and no unit ("Section 3 L.").
	labelWords = set("section", "chapter", "part", "page", "line", "verse", "figure", "table", "item", "row", "room",
		"grade", "step", "rule", "article", "no", "number")
	// Words ending in "s" that are no plural nouns, so no count follows a number before them ("2021-3456 was").
	notPluralNouns = set("is", "was", "has", "does", "as", "this", "his", "its", "us", "yes", "plus", "thus", "less",
		"unless", "across", "always", "perhaps", "whereas", "towards", "afterwards", "besides")
	// A noun after a number that makes it a count, not an identifier ("245-6789 sheep", "confirmation 3-4 days").
	identifierCountNouns = set("people", "men", "women", "children", "sheep", "cattle", "days", "weeks", "months",
		"years", "hours", "minutes", "seconds", "times", "items", "pages", "copies", "miles", "yards", "feet", "acres",
		"dollars", "pounds", "units", "pieces", "persons", "soldiers", "horses", "ships")
	clauseEndPattern = compile(`[.;!?]`)
)

type identifierKind int

const (
	notIdentifier identifierKind = iota
	strongIdentifier
	referenceIdentifier
	legalIdentifier
	phoneIdentifier
)

// labelBefore is the word right before start in lowercase, or the one before a "No." or "#" there, and whether such
// a "No." or "#" stands there; ("", false) when the number starts the text or follows punctuation.
func labelBefore(text string, start int) (string, bool) {
	label, ok := labelPattern.find(before(text, start, 60))
	if !ok {
		return "", false
	}

	word := strings.ToLower(strings.ReplaceAll(label.group(1), "/", ""))
	slot := label.group(2)
	// "number" is the label's slot only after a label word ("Order number 5"), not in prose ("in number 3000").
	numbered := slot != "" && (!strings.HasPrefix(strings.ToLower(slot), "number") || identifierWordsSet[word])
	return word, numbered
}

// identifierLabel is the kind the word right before start (or its "No.") gives a number.
func identifierLabel(text string, start int) identifierKind {
	word, numbered := labelBefore(text, start)
	switch {
	case legalLabels[word]:
		return legalIdentifier
	case strongLabels[word]:
		return strongIdentifier
	case referenceLabels[word]:
		return referenceIdentifier
	case phoneLabels[word]:
		return phoneIdentifier
	case numbered:
		return referenceIdentifier
	}
	return notIdentifier
}

// yearRange is whether a hyphened pair is a range of years ("1820-1830"), read as years before any label.
func yearRange(start, end string) bool {
	return len(start) == 4 && len(end) == 4 && isYear(number(start)) && isYear(number(end)) &&
		number(start) < number(end)
}

// phonePair is whether a hyphened pair has a local phone number's shape ("555-0199"): an exchange that starts from
// two up and a four-digit line number that is not round.
func phonePair(start, end string) bool {
	return len(start) == 3 && len(end) == 4 && strings.ContainsRune("23456789", rune(start[0])) &&
		!strings.HasSuffix(end, "00")
}

// hyphenedIdentifier is the kind of identifier a plain hyphened number ("2021-3456") is, read part by part, or
// notIdentifier to leave it a range: only a strong label, "No." or "#" right before it makes one, and a phone label
// only for a phone-shaped pair.
func hyphenedIdentifier(text string, start, end int, parts []string) identifierKind {
	if len(parts) == 2 && yearRange(parts[0], parts[1]) {
		return notIdentifier
	}

	label := identifierLabel(text, start)
	counted := countedAfter(text, end)
	// A strong label needs an identifier's shape after it, not a count ("confirmation 3-4 days"), and any label more
	// than two short parts ("Lot 12-14").
	if slices.ContainsFunc(parts, func(part string) bool { return len(part) > 2 }) {
		_, numbered := labelBefore(text, start)
		if ((label == legalIdentifier || label == strongIdentifier) && !counted) ||
			(label == referenceIdentifier && numbered) {
			return label
		}
	}

	phoneLabel := phoneBeforePattern.MatchString(before(text, start, 30))
	if len(parts) == 2 && phonePair(parts[0], parts[1]) && phoneLabel && !counted {
		return phoneIdentifier
	}
	return notIdentifier
}

// countedAfter is whether a count noun follows the number ending at end ("3-4 days", "245-6789 sheep").
func countedAfter(text string, end int) bool {
	return identifierCountNouns[lowerWordAfterPattern.firstGroup(text[end:])]
}

// cricketTotal is whether a hyphened pair is a cricket total: a cricket word in its own clause ("England were 250-7
// at stumps after the second innings").
func cricketTotal(text string, start, end int) bool {
	clausesBefore := clauseEndPattern.Split(before(text, start, 40), -1)
	clausesAfter := clauseEndPattern.Split(after(text, end, 40), -1)
	return cricketPattern.MatchString(clausesBefore[len(clausesBefore)-1] + " " + clausesAfter[0])
}

// longDigitRun is whether an unbroken run of digits is too long for a quantity ("12345678" is a barcode or account
// number); a round number stays a quantity ("3000000000").
func longDigitRun(digits string) bool {
	return len(digits) >= 8 && strings.Trim(digits[2:], "0") != ""
}
