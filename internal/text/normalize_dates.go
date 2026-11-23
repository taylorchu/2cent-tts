package text

import (
	"strconv"
	"strings"
)

var (
	monthNames = []string{"January", "February", "March", "April", "May", "June", "July", "August",
		"September", "October", "November", "December"}
	monthAbbreviations     = monthAbbreviationTable()
	monthAbbreviationNames = longestFirst(sortedKeys(monthAbbreviations))
	dottedDatePattern      = compile(`\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b`)
	// A hyphen between a number and a new sentence is a dash ("Sept. 28-We met").
	gluedDashPattern       = compile(`(\d)-([A-Z][a-z]+)\b`)
	hyphenedNumbersPattern = compile(`\b` + numberShape + `(?:-` + numberShape + `)+\b`)
	slashedNumbersPattern  = compile(`\b(\d{1,4})/(\d{1,4})(?:/(\d{1,4}))?\b`)
	// A score out of 5, 10 or 100 ("4.5/5", "rated it 8/10", "10/10 would go again").
	ratingPattern       = compile(`\b(\d{1,3}(?:\.\d)?)/(5|10|100)\b`)
	ratingBeforePattern = compile(`\b(?i:rated?|rating|ratings|scored?|scores|gave(?: it)?|give(?: it)?|gets?|got) $`)
	ratingAfterPattern  = compile(`^ ?(?:\x{2b50}|\x{2605}|(?i:stars?|would|recommend|overall)\b)`)
	// A word right before a slashed pair that makes it a date even where it could be a fraction ("on 3/4 he left").
	dateWordBeforePattern = compile(
		`\b(?i:on|until|till|arriving|arrives|due|shipped|delivered|since|dated|expires?) $`)
	// Denominators of measuring fractions, never days ("3/16 inch").
	fractionDenominators = map[int]bool{16: true, 32: true, 64: true}
	expiryPattern        = compile(`(?i:expires?|expiry|exp\.?|valid (?:thru|until))(?: on| date:?)? $`)
	colonNumbersPattern  = compile(`\b(\d+):(\d+(?:,\d{3})*)\b`)
	// A ratio or a duration needs a word that says so; any other pair that is no time ("25:5 And it came to pass")
	// is a verse or reference read as two numbers.
	ratioWordsPattern = compile(`\b(?i:ratio|odds|scale|mix|aspect|bets?|score|majority|screen|display|monitor` +
		`|format|widescreen)\b`)
	durationWordsPattern = compile(`\b(?i:lasted|lasts|took|takes|long|minutes?|seconds?|duration|length|recording` +
		`|song|track|video|clip)\b`)
	colonBetweenNumbersPattern = compile(`(\d):(\d)`)
	dateDeterminerPattern      = compile(`\b(?i:the|this|that|a|an|every|each|next|last|his|her|its|our|their|my` +
		`|your) $|\d(?:st|nd|rd|th|d) (?:and|or) $`)
	dayMonthPattern = compile(`\b(\d{1,2})` + ordinalSuffix + `? (?:of )?([A-Za-z]{3,9})\b(\.?)`)
	monthDayPattern = compile(`\b([A-Za-z]{3,9})\.? (\d{1,2})` + ordinalSuffix + `?\b`)

	monthAfterPattern       = compile(`^ ([A-Za-z]{3,9})\b`)
	monthBeforePattern      = compile(`\b([A-Za-z]{3,9}) $`)
	betweenBeforePattern    = compile(`\b[Bb]etween $`)
	digitBeforePattern      = compile(`\d $`)
	slashAfterPattern       = compile(`^[A-Za-z0-9/]`)
	pluralAfterPattern      = compile(`^ ([a-z]+s)\b`)
	ofAfterPattern          = compile(`^ of\b`)
	colonDigitPattern       = compile(`^:\d`)
	dayDigitAfterPattern    = compile(`^[:.]?\d`)
	inBeforePattern         = compile(`\b[Ii]n $`)
	commaOrYearAfterPattern = compile(`^(?:,| \d{4})`)
)

// dottedDate reads "15.11.2026" as "15 November 2026"; a date written with periods is day first.
func dottedDate(m match) (string, bool) {
	day, month, year := number(m.group(1)), number(m.group(2)), number(m.group(3))
	if continuesNumber(m) || !isDayOfMonth(day) || !isMonth(month) {
		return "", false
	}
	return dayFirstDate(day, month) + " " + strconv.Itoa(year), true
}

// yearText keeps a four-digit year as written and reads "80" as "eighty" and "05" as "oh five".
func yearText(year string) string {
	if len(year) == 4 {
		return year
	}
	return twoDigitWords(year)
}

// numericDate reads a date written in numbers ("04-09-1980", "2024/03/15", "4-9-80") as "April 9 1980", month first
// unless dayFirst or only day first is valid ("23-04-1980" as "23 April 1980"), or declines when no date fits.
func numericDate(parts []string, dayFirst bool) (string, bool) {
	if len(parts[0]) == 4 {
		return yearFirstDate(parts[0], parts[1], parts[2])
	}

	first, second, year := parts[0], parts[1], parts[2]
	if len(first) > 2 || len(second) > 2 || (len(year) != 2 && len(year) != 4) {
		return "", false
	}

	date, ok := slashedDate(number(first), number(second), dayFirst)
	if !ok {
		return "", false
	}
	return date + " " + yearText(year), true
}

// yearFirstDate reads "2024/03/15" as "March 15 2024", or declines when the month or day is out of range.
func yearFirstDate(year, month, day string) (string, bool) {
	if len(month) <= 2 && len(day) <= 2 && isMonth(number(month)) && isDayOfMonth(number(day)) {
		return monthFirstDate(number(month), number(day)) + " " + year, true
	}
	return "", false
}

// hyphenedNumbers reads numbers joined by hyphens: an identifier read part by part, an ISO date, a date range, a
// cricket total ("250-7" as "250 for 7"), or a range read with "to" ("3-5").
func hyphenedNumbers(m match) (string, bool) {
	// A range after a colon is a range of verses ("Matthew 5:3-12").
	if continuesAfter(m) || (continuesBefore(m) && previousCharacter(m) != ':') {
		return "", false
	}

	parts := strings.Split(m.group(0), "-")
	plain := allDigits(parts)
	if date, ok := hyphenedDate(m, parts, plain); ok {
		return date, true
	}

	identifier := notIdentifier
	if plain {
		identifier = hyphenedIdentifier(m.text, m.start(), m.end(), parts)
	}

	if identifier != notIdentifier || len(parts) >= 3 {
		if !plain {
			return "", false
		}
		return identifierWords(m.group(0), identifier == legalIdentifier || identifier == strongIdentifier, false), true
	}

	start, end := parts[0], parts[1]
	// A range of years reads both ends as years, before any count after it ("1840-1850 copies").
	if plain && yearRange(start, end) && !continuesBefore(m) {
		return rangeWords(m, yearWords(number(start)), yearWords(number(end))), true
	}

	if plain {
		return hyphenedPair(m, start, end), true
	}
	return rangeWords(m, start, end), true
}

// hyphenedDate reads three plain parts as a date where no identifier label stands before them.
func hyphenedDate(m match, parts []string, plain bool) (string, bool) {
	if len(parts) != 3 || !plain {
		return "", false
	}

	date, ok := numericDate(parts, false)
	if !ok || identifierLabel(m.text, m.start()) != notIdentifier {
		return "", false
	}
	return date, true
}

// hyphenedPair reads two whole numbers joined by a hyphen as a phone or reference number, a date range or a range.
func hyphenedPair(m match, start, end string) string {
	leadingZero := (len(start) > 1 && start[0] == '0') || (len(end) > 1 && end[0] == '0')
	phoneShaped := max(len(start), len(end)) >= 5 || longPair(start, end)
	label := identifierLabel(m.text, m.start())
	if leadingZero || (phoneShaped && label != notIdentifier) {
		return identifierWords(m.group(0), label == notIdentifier, false)
	}

	if start == end {
		return start + " " + end
	}

	if compareNumbers(start, end) > 0 && number(end) <= 10 && cricketTotal(m.text, m.start(), m.end()) {
		return start + " for " + end
	}

	if number(start) <= 31 && number(end) <= 31 {
		if days, ok := ordinalDayRange(m, start, end); ok {
			return days
		}
	}
	return rangeWords(m, start, end)
}

// longPair is whether both numbers of a pair have four digits or more and are not a pair of years or of round hundreds.
func longPair(start, end string) bool {
	years := isYear(number(start)) && isYear(number(end))
	roundHundreds := strings.HasSuffix(start, "00") && strings.HasSuffix(end, "00")
	return min(len(start), len(end)) >= 4 && !years && !roundHundreds
}

// ordinalDayRange reads a pair of days before or after a month name as ordinals ("the 3rd to 5 May", "May 3rd to 5th").
func ordinalDayRange(m match, start, end string) (string, bool) {
	if monthName(monthAfterPattern.firstGroup(textAfter(m))) != "" {
		return theBefore(m) + start + ordinalSuffixOf(number(start)) + " to " + end, true
	}

	if month, ok := monthBeforePattern.find(textBefore(m)); ok && monthName(month.group(1)) != "" {
		return start + ordinalSuffixOf(number(start)) + " to " + end + ordinalSuffixOf(number(end)), true
	}
	return "", false
}

// gluedDash sets a hyphen joining a number to a sentence starter apart as a dash, and declines for a compound
// ("4-Hour").
func gluedDash(m match) (string, bool) {
	if !sentenceStarters[m.group(2)] {
		return "", false
	}
	return m.group(1) + " - " + m.group(2), true
}

// rangeWords reads a range with "to", or with "and" after "between" ("between 10000-20000 men").
func rangeWords(m match, start, end string) string {
	if betweenBeforePattern.MatchString(textBefore(m)) {
		return start + " and " + end
	}
	return start + " to " + end
}

// slashedDate reads a day and month written in numbers, month first unless dayFirst or only day first is valid, or
// declines when neither order is a date.
func slashedDate(first, second int, dayFirst bool) (string, bool) {
	if isMonth(second) && (first > 12 || dayFirst) && isDayOfMonth(first) {
		return dayFirstDate(first, second), true
	}

	if isMonth(first) && isDayOfMonth(second) {
		return monthFirstDate(first, second), true
	}
	return "", false
}

// slashedYearDate reads "12/10/2026" as "December 10 2026", and a slash-separated reference that is no date digit by
// digit.
func slashedYearDate(first, second, third string, dayFirst bool) string {
	if date, ok := numericDate([]string{first, second, third}, dayFirst); ok {
		return date
	}
	return joined([]string{first, second, third}, " slash ", digitWords)
}

// slashedFraction reads "3/4" and "1 3/4" as fractions, "24/7" and "9/11" as said, and an improper pair as "over".
func slashedFraction(first, second, preceding string) string {
	firstNumber, secondNumber := number(first), number(second)
	if (first == "24" && second == "7") || (first == "9" && second == "11") {
		return integerWords(firstNumber) + " " + integerWords(secondNumber)
	}

	proper := 0 < firstNumber && firstNumber < secondNumber
	if proper && digitBeforePattern.MatchString(preceding) {
		return "and " + fractionWords(strconv.Itoa(firstNumber), strconv.Itoa(secondNumber), true)
	}

	if firstNumber == secondNumber {
		return first + " " + second
	}

	if proper {
		return fractionWords(strconv.Itoa(firstNumber), strconv.Itoa(secondNumber), false)
	}
	return first + " over " + second
}

// rating reads a score out of 5, 10 or 100 as "4.5 out of 5": a decimal score, a full score ("10/10"), or one with a
// rating word before or after it. A date word before keeps it a date ("due 5/5").
func rating(m match) (string, bool) {
	score, scale := m.group(1), m.group(2)
	previous, preceding, following := previousCharacter(m), textBefore(m), textAfter(m)
	scoreValue, _ := strconv.ParseFloat(score, 64)
	if isAlphanumeric(previous) || strings.ContainsRune("/.-$", previous) || slashAfterPattern.MatchString(following) ||
		scoreValue > float64(number(scale)) || dateWordBeforePattern.MatchString(preceding) {
		return "", false
	}

	rated := strings.Contains(score, ".") || score == scale || ratingBeforePattern.MatchString(preceding) ||
		ratingAfterPattern.MatchString(following)
	if !rated {
		return "", false
	}
	return score + " out of " + scale, true
}

// slashedNumbers reads "31/12" as "31 December" and "12/10" or "3/14" as a month and day (month first unless the
// text writes day first); a fraction ("3/4", "3/16 inch") in words; a slashed reference ("2026/0917/44") digit by
// digit.
func slashedNumbers(m match, dayFirst bool) (string, bool) {
	first, second, third := m.group(1), m.group(2), m.group(3)
	previous := previousCharacter(m)
	if isAlphanumeric(previous) || strings.ContainsRune("/.-", previous) ||
		slashAfterPattern.MatchString(textAfter(m)) {
		return "", false
	}

	if third != "" {
		return slashedYearDate(first, second, third, dayFirst), true
	}

	if max(len(first), len(second)) >= 3 {
		return digitWords(first) + " slash " + digitWords(second), true
	}

	following, preceding := textAfter(m), textBefore(m)
	// Before a counted noun or "of" a slashed pair is a fraction ("won 11/12 games", "3/4 of the class").
	isCounted := slashedCounted(first, following)
	firstNumber, secondNumber := number(first), number(second)
	if len(first) == 2 && len(second) == 2 && !isCounted {
		return slashedTwoDigits(first, second, preceding, dayFirst), true
	}

	dateShaped := slashedDateShaped(firstNumber, secondNumber, preceding)
	aroundTheClock := first == "24" && second == "7"
	if dateShaped && !isCounted && !digitBeforePattern.MatchString(preceding) && !aroundTheClock {
		if date, ok := slashedDate(firstNumber, secondNumber, dayFirst); ok {
			return date, true
		}
	}
	return slashedFraction(first, second, preceding), true
}

// slashedCounted is whether a slashed pair is a fraction before a counted noun or "of".
func slashedCounted(first, following string) bool {
	plural, isPlural := pluralAfterPattern.find(following)
	return ofAfterPattern.MatchString(following) || (isPlural && !notPluralNouns[plural.group(1)] && first[0] != '0')
}

// slashedTwoDigits reads a pair of two-digit numbers as a card's expiry month, a date, or a fraction.
func slashedTwoDigits(first, second, preceding string, dayFirst bool) string {
	firstNumber, secondNumber := number(first), number(second)
	if expiryPattern.MatchString(preceding) && isMonth(firstNumber) {
		return monthNames[firstNumber-1] + " 20" + second
	}

	if date, ok := slashedDate(firstNumber, secondNumber, dayFirst); ok {
		return date
	}
	return slashedFraction(first, second, preceding)
}

func slashedDateShaped(firstNumber, secondNumber int, preceding string) bool {
	// A pair that is no everyday fraction is a date ("on 3/14", "7/4"): its day passes twelve, or it is improper.
	return firstNumber <= 12 && (secondNumber > 12 || firstNumber > secondNumber ||
		dateWordBeforePattern.MatchString(preceding)) && !fractionDenominators[secondNumber]
}

// colonNumbers reads a ratio ("16:9") with "to" and a duration past a day's hours ("48:12") in minutes and seconds,
// each where a nearby word says so; any other pair that is no time ("Isaiah 40:31") as its two numbers with a pause.
func colonNumbers(m match) (string, bool) {
	first, second := m.group(1), m.group(2)
	if previousCharacter(m) == ':' || colonDigitPattern.MatchString(textAfter(m)) {
		return "", false
	}

	if len(first) <= 2 && len(second) == 2 && number(first) <= 23 {
		return "", false
	}

	wordsBefore := strings.Fields(textBefore(m))
	wordsAfter := strings.Fields(textAfter(m))
	nearby := strings.Join(append(wordsBefore[max(0, len(wordsBefore)-4):], wordsAfter[:min(2, len(wordsAfter))]...),
		" ")
	if (strings.Contains(second, ",") || len(second) == 1) && ratioWordsPattern.MatchString(nearby) {
		return first + " to " + second, true
	}

	if len(second) == 2 && number(second) <= 59 && durationWordsPattern.MatchString(nearby) {
		return first + " minutes " + second + " seconds", true
	}
	return first + ", " + second, true
}

func isMonth(value int) bool {
	return 1 <= value && value <= 12
}

func isDayOfMonth(value int) bool {
	return 1 <= value && value <= 31
}

// monthFirstDate writes a date as "March 14", for the date rules to read.
func monthFirstDate(month, day int) string {
	return monthNames[month-1] + " " + strconv.Itoa(day)
}

// dayFirstDate writes a date as "14 March", for the date rules to read.
func dayFirstDate(day, month int) string {
	return strconv.Itoa(day) + " " + monthNames[month-1]
}

// monthName is the full name of the month word names ("FEB" as "February"), or ""; a lowercase word is not a month
// ("they march 3 abreast").
func monthName(word string) string {
	if !isTitle(word) && !isUpper(word) {
		return ""
	}

	titled := capitalize(word)
	for _, name := range monthNames {
		if name == titled {
			return name
		}
	}
	return monthAbbreviations[titled]
}

// theBefore is "the " before an ordinal date, unless a determiner already stands there ("this 1st of May").
func theBefore(m match) string {
	if dateDeterminerPattern.MatchString(textBefore(m)) {
		return ""
	}
	return "the "
}

// dayMonth reads "14 March" as "the fourteenth of March".
func dayMonth(m match) (string, bool) {
	day, month, period := m.group(1), m.group(2), m.group(3)
	name := monthName(month)
	if name == "" || !isDayOfMonth(number(day)) {
		return "", false
	}

	// The period of an abbreviated month ("Jan.") stays only where it also ends the sentence.
	if name != capitalize(month) && !sentenceEnd(m.text, m.end()) {
		period = ""
	}

	// A day-first date always says "the" unless a determiner stands there ("15th August", "this 1st of May").
	return theBefore(m) + ordinalWords(integerWords(number(day))) + " of " + name + period, true
}

// monthDay reads "March 14" as "March fourteenth".
func monthDay(m match) (string, bool) {
	month, day := m.group(1), m.group(2)
	name := monthName(month)
	following := textAfter(m)
	if name == "" || !isDayOfMonth(number(day)) || dayDigitAfterPattern.MatchString(following) {
		return "", false
	}

	// "In May 5 men died" is a count after the month, not a date.
	whole := m.group(0)
	if inBeforePattern.MatchString(textBefore(m)) && isDigits(whole[len(whole)-1:]) &&
		!commaOrYearAfterPattern.MatchString(following) {
		return "", false
	}
	return name + " " + ordinalWords(integerWords(number(day))), true
}

// monthAbbreviationTable maps each abbreviated month to its name: its first three letters, May aside, and "Sept".
func monthAbbreviationTable() map[string]string {
	abbreviations := map[string]string{}
	for _, name := range monthNames {
		if name != "May" {
			abbreviations[name[:3]] = name
		}
	}

	abbreviations["Sept"] = abbreviations["Sep"]
	return abbreviations
}
