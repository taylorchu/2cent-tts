package text

import (
	"strconv"
	"strings"
)

var (
	bookNumbers         = map[string]string{"1": "First", "2": "Second", "3": "Third"}
	numberedBookPattern = compile(`\b([123]) (Samuel|Kings|Chronicles|Corinthians|Thessalonians|Timothy|Peter|John) ` +
		chapterAndVerse)

	numberAbbreviations = map[string]string{"no": "number", "fig": "figure", "pp": "pages", "p": "page",
		"ch": "chapter", "sc": "scene", "lat": "latitude", "long": "longitude"}
	numberAbbreviationPattern = compile(`\b((?i:` +
		longestFirst(sortedKeys(numberAbbreviations)) +
		`))\. ((?:is )?)(\d+|[ivxlc]+\b)(?:-([ivxlc]+)\b)?`)
	versusPattern = compile(`\b([A-Z][a-z]+) v\. ([A-Z][a-z])|\bvs\. `)
	// A number after "chap." or "vol." ("Vol. X of the records"); "I" needs punctuation, "of" or "and" after it,
	// since "chap. I think" is the pronoun.
	numeralAfterPattern = compile(`^ (?:\d|[IVXLCivxlc]{2,}\b|[VXvx]\b|[IVXLCivxlc](?:[.,;:)]|$| of\b| and\b))`)
	// A sentence starter of two or more letters that is no title-case name ("the Lt. He was brave", not "the Rev. A
	// Tragedy of Twopence").
	newSentenceAfterPattern   = compile(`^ ([A-Z][a-z]+)\b( [A-Z])?`)
	bracketedCapitalPattern   = compile(`^\s+[(\[][A-Z]`)
	titleWithoutPeriodPattern = compile(`\b(Mrs|Mr|Ms|Dr|St) ([A-Z][a-z])`)
	// Latin abbreviations read by meaning.
	latinAbbreviations = map[string]string{"i.e.": "that is", "e.g.": "for example", "cf.": "compare",
		"et al.": "and others", "viz.": "namely"}
	latinPattern = compile(`\b(` + longestFirst(sortedKeys(latinAbbreviations)) + `)`)
	// Country initialisms said letter by letter ("the USA"); "US" only after "the", since capitals set "US" as a word
	// ("LET US GO").
	spelledInitialismPattern = compile(`\b((?:[Tt]he )?)(USA|UK|UAE|EU|USSR|US)\b`)
	// A title that ends a signature before a new sentence ("U. S. GRANT, Lt. Gen. When I").
	signatureBeforePattern = compile(`[A-Z][A-Za-z]*, (?:[A-Z][A-Za-z]*\. )?$`)
	// A title used as a noun after "the" or "my", or after another title ("Ask the Dr. about it", "The Rt. Hon.
	// gentleman").
	titleNounBeforePattern = compile(`\b(?i:the|my|his|her|our|your|their|a|an)(?: [A-Z][a-z]*\.)? $`)
	// A county before a state ("Fauquier Co., Va.") or after a sovereign's name ("Queen's Co.").
	countyPattern       = compile(`^, (?:` + oldStateNames + `)\.`)
	countyBeforePattern = compile(`\b(?:Queen|King)'s $`)
	streetBeforePattern = compile(`([A-Z][a-z]+|\d+(?:st|nd|rd|th)) $`)

	letterPeriodPattern     = compile(`^[A-Za-z]\.`)
	wordEndBeforePattern    = compile(`[A-Za-z]\.$|[A-Za-z]['-]$`)
	numberAfterDotPattern   = compile(`^[.,]\d`)
	codeAfterPattern        = compile(`^[-/][A-Za-z0-9]`)
	nextCapitalizedPattern  = compile(`^ [\"']?([A-Z][A-Za-z]*)`)
	lowerAfterPattern       = compile(`^ [a-z]`)
	firmBeforePattern       = compile(`(?:&|\band|[A-Z][A-Za-z']*\.?) $`)
	referenceAfterPattern   = compile(`^ (?:\d|#|no\b|number\b)`)
	ordinalBeforePattern    = compile(`\d(?:st|nd|rd|th|d) $`)
	nameBeforePattern       = compile(`[A-Z][a-z]*,? $`)
	firmNameBeforePattern   = compile(`([A-Z][A-Za-z]*(?:'s|\.)?) $`)
	countyNameBeforePattern = compile(`[A-Z][a-z]+ $`)
	ampersandBeforePattern  = compile(`(?:&|\band) $`)
	titlesAfterPattern      = compile(`^ and M(?:rs?|s)\.`)
	abbreviations           = abbreviationTable()
	titles                  = readingTable(abbreviations, func(row abbreviation) string { return row.title })
	streets                 = readingTable(abbreviations, func(row abbreviation) string { return row.street })
	// A title as it is said ("Missus Dr. dear") names no street, so text already in spoken form reads as it did.
	spokenTitles             = union(set(mapValues(titles)...), set("Miss"))
	streetWords              = union(set(mapValues(streets)...), set("Way", "Row", "Alley", "Plaza", "Trail"))
	abbreviationWordsPattern = compile(`\b(` + longestFirst(abbreviationNames(abbreviations)) + `)\.`)
)

// chapterAndVerse is the chapter and verse after a Bible book: "1:3", or a Roman chapter and a verse ("xv. 11").
const chapterAndVerse = `(\d+:\d+|[ivxlc]+\.? \d+)`

// periodPolicy is how a word abbreviation keeps its period: where the sentence ends, there unless a number follows
// ("Chap. IV.", "the second vol. He"), or also before a bracket opening a capitalized word, since "etc." mostly ends
// its sentence ("etc. etc. (Imported soft").
type periodPolicy int

const (
	periodAtSentenceEnd periodPolicy = iota
	periodUnlessNumbered
	periodAtSentenceOrBracket
)

// abbreviation is one abbreviation and its readings, tried in this order: post-name (a title after a name, "John
// Smith, Esq."), county ("Co. Cork", or a company of soldiers, "Co. A"), street ("Main St."), title ("Dr. Brown") and
// word ("approx."). needs is a context the abbreviation must have for any reading; nameOrPlace marks a title that is
// also a place or a given name, so its period may end a sentence ("the family of Col. He"); notAfterThe marks a
// title never read as a noun after "the" ("the Gen." in a grammar is the genitive).
type abbreviation struct {
	title, street, postName, county, word string
	nameOrPlace, notAfterThe              bool
	period                                periodPolicy
	needs                                 func(spot) bool
}

func abbreviationTable() map[string]abbreviation {
	return map[string]abbreviation{
		"Mr":  {title: "Mister"},
		"Mrs": {title: "Missus"},
		// Readers say "Messrs." as "messers".
		"Messrs": {title: "Messers"},
		"Ms":     {title: "Miz"},
		"Mme":    {title: "Madame"},
		"Mlle":   {title: "Mademoiselle"},
		"Dr":     {title: "Doctor", street: "Drive"},
		"St":     {title: "Saint", street: "Street"},
		"Mt":     {title: "Mount"},
		"Ft":     {title: "Fort"},
		"Prof":   {title: "Professor"},
		"Rev":    {title: "Reverend"},
		"Hon":    {title: "Honorable"},
		"Rt":     {title: "Right"},
		"Gov":    {title: "Governor"},
		"Supt":   {title: "Superintendent"},
		"Insp":   {title: "Inspector"},
		"Sgt":    {title: "Sergeant"},
		"Sergt":  {title: "Sergeant"},
		"Capt":   {title: "Captain"},
		"Lt":     {title: "Lieutenant"},
		"Lieut":  {title: "Lieutenant"},
		"Maj":    {title: "Major"},
		"Col":    {title: "Colonel", nameOrPlace: true},
		"Gen":    {title: "General", nameOrPlace: true, notAfterThe: true},
		"Wm":     {title: "William", nameOrPlace: true},
		"Geo":    {title: "George", nameOrPlace: true},
		"Thos":   {title: "Thomas", nameOrPlace: true},
		"Chas":   {title: "Charles", nameOrPlace: true},
		"Jas":    {title: "James", nameOrPlace: true},
		"Benj":   {title: "Benjamin", nameOrPlace: true},
		"Jno":    {title: "John", nameOrPlace: true},
		"Ave":    {street: "Avenue"},
		"Rd":     {street: "Road"},
		"Blvd":   {street: "Boulevard"},
		"Ln":     {street: "Lane"},
		"Ct":     {street: "Court"},
		"Pl":     {street: "Place"},
		"Pkwy":   {street: "Parkway"},
		"Hwy":    {street: "Highway"},
		"Sq":     {street: "Square"},
		"Ter":    {street: "Terrace"},
		"Cir":    {street: "Circle"},
		"Esq":    {postName: "Esquire"},
		"Jr":     {postName: "Junior"},
		"Sr":     {postName: "Senior"},
		"approx": {word: "approximately"},
		"dept":   {word: "department"},
		"max":    {word: "maximum", needs: lowerOrBeforeLower},
		"est":    {word: "estimated", needs: lowerOrBeforeLower},
		"ref":    {word: "reference", needs: numberAfter},
		"fl":     {word: "floor", needs: ordinalBefore},
		"Co":     {county: "County", word: "company"},
		// A lowercase "co." elsewhere than after a firm is Latin ("ammon. co.").
		"co":   {word: "company", needs: firmBefore},
		"etc":  {word: "etcetera", period: periodAtSentenceOrBracket},
		"incl": {word: "including"},
		"rm":   {word: "room", needs: numberAfter},
		"sp":   {word: "species", needs: genusBefore},
		"vol":  {word: "volume", period: periodUnlessNumbered},
		"chap": {word: "chapter", period: periodUnlessNumbered, needs: numeralAfter},
		"inst": {word: "instant", needs: ordinalBefore},
		"ult":  {word: "ultimo", needs: ordinalBefore},
		"prox": {word: "proximo", needs: ordinalBefore},
	}
}

// readingTable maps each abbreviation that has a reading to it.
func readingTable(table map[string]abbreviation, reading func(abbreviation) string) map[string]string {
	found := map[string]string{}
	for name, row := range table {
		if spoken := reading(row); spoken != "" {
			found[name] = spoken
		}
	}
	return found
}

// abbreviationNames are the forms the table matches: titles and streets also in capitals ("ST. LOUIS"), and words
// also capitalized.
func abbreviationNames(table map[string]abbreviation) []string {
	names := map[string]bool{}
	for name, row := range table {
		names[name] = true
		if (row.title != "" || row.street != "") && len(name) > 1 {
			names[strings.ToUpper(name)] = true
		}

		if row.word != "" {
			names[capitalize(name)] = true
		}
	}
	return sortedKeys(names)
}

func numberedBook(m match) string {
	return bookNumbers[m.group(1)] + " " + m.group(2) + " " + m.group(3)
}

func versus(m match) (string, bool) {
	if m.group(1) != "" {
		return m.group(1) + " versus " + m.group(2), true
	}
	return "versus ", true
}

// period is the period an expanded abbreviation ending at end keeps under policy: "." or "".
func period(text string, end int, policy periodPolicy) string {
	if policy == periodUnlessNumbered && numeralAfterPattern.MatchString(text[end:]) {
		return ""
	}

	bracketed := policy == periodAtSentenceOrBracket && bracketedCapitalPattern.MatchString(text[end:])
	if sentenceEnd(text, end) || bracketed {
		return "."
	}
	return ""
}

// numberAbbreviation reads "no. 7" as "number 7", "p. 45" as "page 45" and "ch. iii" as "chapter 3"; a long number
// after "no." is a reference read digit by digit, and a hyphened one is left to the identifier reader ("No.
// 2021-CV-0042").
func numberAbbreviation(m match) (string, bool) {
	written, verb, numeral, last := m.group(1), m.group(2), m.group(3), m.group(4)
	word := numberAbbreviations[strings.ToLower(written)]
	if isUpper(written[:1]) {
		word = capitalize(word)
	}

	following := textAfter(m)
	if numberAfterDotPattern.MatchString(following) {
		return "", false
	}

	// A range of Roman pages reads both ends ("pp. iv-x" as "pages 4 to 10").
	if last != "" {
		first, firstValid := romanValue(numeral)
		end, endValid := romanValue(last)
		if isDigits(numeral) || !firstValid || !endValid {
			return "", false
		}
		return word + " " + verb + strconv.Itoa(first) + " to " + strconv.Itoa(end), true
	}

	if codeAfterPattern.MatchString(following) {
		return word + " " + verb + numeral, true
	}

	if !isDigits(numeral) {
		value, ok := romanValue(numeral)
		if !ok {
			return "", false
		}
		numeral = strconv.Itoa(value)
	}

	if strings.EqualFold(written, "no") && len(numeral) >= 4 {
		numeral = digitWords(numeral)
	}
	return word + " " + verb + numeral, true
}

// spokenLike writes word in capitals where written is in capitals ("ST." as "SAINT").
func spokenLike(written, word string) string {
	if isUpper(written) && len(written) > 1 {
		return strings.ToUpper(word)
	}
	return word
}

// nextCapitalized is the capitalized word after the abbreviation, past an opening quote, or "".
func nextCapitalized(s spot) string {
	return nextCapitalizedPattern.firstGroup(s.following)
}

// lowerOrBeforeLower is whether "max."/"est." is lowercase, or capitalized before a lowercase word, since "Max." alone
// is a name.
func lowerOrBeforeLower(s spot) bool {
	return !isUpper(s.group(1)[:1]) || lowerAfterPattern.MatchString(s.following)
}

// genusBefore is whether a lowercase "sp." follows a genus, a capitalized word that starts no sentence ("Aspergillus
// sp.").
func genusBefore(s spot) bool {
	genus, ok := capitalizedBeforePattern.find(s.preceding)
	return isLower(s.group(1)) && ok && !sentenceStarters[genus.group(1)]
}

// firmBefore is whether "co." follows a firm's name or "&" ("Smith & co.").
func firmBefore(s spot) bool {
	return firmBeforePattern.MatchString(s.preceding)
}

func numberAfter(s spot) bool {
	return referenceAfterPattern.MatchString(s.following)
}

func ordinalBefore(s spot) bool {
	return ordinalBeforePattern.MatchString(s.preceding)
}

func numeralAfter(s spot) bool {
	return numeralAfterPattern.MatchString(s.following)
}

// readingResult tells a reading that applies (spoken, or declined to keep the text as written) from one that does
// not, after which the next reading is tried.
type readingResult int

const (
	readingSpoken readingResult = iota
	readingDeclined
	readingSkipped
)

// postNameReading reads a title after a name ("Smith, Esq.").
func postNameReading(row abbreviation, s spot) (string, readingResult) {
	if row.postName == "" {
		return "", readingSkipped
	}

	if !nameBeforePattern.MatchString(s.preceding) {
		return "", readingDeclined
	}
	return row.postName + sentencePeriod(s.text, s.end()), readingSpoken
}

// countyReading reads "Co." as a company of soldiers ("Co. A") or a county ("Co. Cork", "Fauquier Co., Va.", "Queen's
// Co."), or skips to read it as the word "Company"; a name or "&" before it makes a firm.
func countyReading(row abbreviation, s spot) (string, readingResult) {
	if row.county == "" {
		return "", readingSkipped
	}

	nextWord := nextCapitalized(s)
	if len(nextWord) == 1 {
		return "Company", readingSpoken
	}

	if (countyPattern.MatchString(s.following) && countyNameBeforePattern.MatchString(s.preceding)) ||
		countyBeforePattern.MatchString(s.preceding) {
		return row.county + sentencePeriod(s.text, s.end()), readingSpoken
	}

	firm, firmFound := firmNameBeforePattern.find(s.preceding)
	if nextWord != "" && !sentenceStarters[nextWord] && !ampersandBeforePattern.MatchString(s.preceding) &&
		(!firmFound || sentenceStarters[firm.group(1)]) {
		return row.county, readingSpoken
	}
	return "", readingSkipped
}

// streetReading reads a street after a street name and before a new sentence or no name ("Main St. The house is
// red"); a hyphen joins "St." to a place name instead ("Chapelle St.-Lambert").
func streetReading(row abbreviation, s spot) (string, readingResult) {
	if row.street == "" {
		return "", readingSkipped
	}

	if row.title != "" && strings.HasPrefix(s.following, "-") {
		return spokenLike(s.group(1), row.title), readingSpoken
	}

	streetBefore, ok := streetBeforePattern.find(s.preceding)
	nextWord := nextCapitalized(s)
	if ok && !sentenceStarters[streetBefore.group(1)] && !spokenTitles[streetBefore.group(1)] &&
		(nextWord == "" || sentenceStarters[nextWord] || row.title == "") {
		return spokenLike(s.group(1), row.street) + sentencePeriod(s.text, s.end()), readingSpoken
	}

	if row.title != "" {
		return "", readingSkipped
	}
	return "", readingDeclined
}

// titleReading reads a title before a name ("Dr. Brown", "Dr. and Mrs. Leete") or after "the" ("Ask the Dr. about
// it"); a title that is also a name or place names no one before a new sentence, unless it ends a signature.
func titleReading(row abbreviation, s spot) (string, readingResult) {
	if row.title == "" {
		return "", readingSkipped
	}

	spokenTitle := spokenLike(s.group(1), row.title)
	next := nextWord(s.text, s.end())
	if row.nameOrPlace && (next == wordStarter || next == wordQuoted) {
		if signatureBeforePattern.MatchString(s.preceding) {
			return spokenTitle + ".", readingSpoken
		}
		return "", readingDeclined
	}

	asNoun := !row.notAfterThe && titleNounBeforePattern.MatchString(s.preceding)
	if asNoun && newSentenceAfter(s.following) && next == wordStarter {
		return spokenTitle + ".", readingSpoken
	}

	if nextCapitalized(s) != "" || titlesAfterPattern.MatchString(s.following) {
		return spokenTitle, readingSpoken
	}

	if asNoun {
		return spokenTitle + sentencePeriod(s.text, s.end()), readingSpoken
	}
	return "", readingDeclined
}

// newSentenceAfter is whether a capitalized word after a title starts a sentence; "An" or "The" before a capital
// does not.
func newSentenceAfter(following string) bool {
	starter, ok := newSentenceAfterPattern.find(following)
	if !ok {
		return false
	}

	article := starter.group(1) == "An" || starter.group(1) == "The"
	return !article || starter.group(2) == ""
}

// wordReading reads a word abbreviation ("approx." as "approximately"), capitalized as written.
func wordReading(row abbreviation, s spot) (string, readingResult) {
	word := row.word
	if isUpper(s.group(1)[:1]) {
		word = capitalize(word)
	}
	return word + period(s.text, s.end(), row.period), readingSpoken
}

var abbreviationReadings = []func(abbreviation, spot) (string, readingResult){
	postNameReading, countyReading, streetReading, titleReading, wordReading,
}

// tableKey is the table row of written: titles and streets by their case, words in lowercase.
func tableKey(written string) string {
	key := written
	if isUpper(written) && len(written) > 1 {
		key = capitalize(written)
	}

	if _, ok := abbreviations[key]; ok {
		return key
	}
	return strings.ToLower(key)
}

// abbreviationWords reads a table abbreviation as the first reading its context allows, or declines to keep it as
// written: a letter and a period after it make an initialism ("M.A.", "vol.i."), and a letter joined by an apostrophe
// or hyphen before it the end of a word ("hon-est.").
func abbreviationWords(m match) (string, bool) {
	s := spotOf(m)
	if letterPeriodPattern.MatchString(s.following) || wordEndBeforePattern.MatchString(s.preceding) {
		return "", false
	}

	row := abbreviations[tableKey(s.group(1))]
	if row.needs != nil && !row.needs(s) {
		return "", false
	}

	for _, reading := range abbreviationReadings {
		spoken, result := reading(row, s)
		if result != readingSkipped {
			return spoken, result == readingSpoken
		}
	}
	return "", false
}

// spelledInitialism reads "the UK" as "the U K".
func spelledInitialism(m match) (string, bool) {
	the, initialism := m.group(1), m.group(2)
	if initialism == "US" && the == "" {
		return "", false
	}
	return the + spelled(initialism), true
}

// latin reads "e.g." as "for example" and "i.e." as "that is"; the period is the abbreviation's, kept only before a
// sentence starter or a quote ("etc., i.e. The"), not at the end of a cut text or before a name or bracket.
func latin(m match) string {
	switch nextWord(m.text, m.end()) {
	case wordStarter, wordQuoted:
		return latinAbbreviations[m.group(1)] + "."
	}
	return latinAbbreviations[m.group(1)]
}
