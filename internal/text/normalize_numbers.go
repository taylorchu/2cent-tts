package text

import (
	"math"
	"strconv"
	"strings"
	"unicode"
)

var (
	digitNames       = strings.Fields("zero one two three four five six seven eight nine")
	smallNumberNames = strings.Fields("zero one two three four five six seven eight nine ten eleven twelve " +
		"thirteen fourteen fifteen sixteen seventeen eighteen nineteen")
	tensNames     = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
	scaleNames    = []string{"", "thousand", "million", "billion", "trillion", "quadrillion"}
	fractionNames = map[string]noun{"2": {"half", "halves"}, "4": {"quarter", "quarters"}}
	ordinalNames  = map[string]string{"one": "first", "two": "second", "three": "third", "five": "fifth",
		"eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
	superscriptDigits = strings.NewReplacer("\u2070", "0", "\u00b9", "1", "\u00b2", "2", "\u00b3", "3", "\u2074", "4",
		"\u2075", "5", "\u2076", "6", "\u2077", "7", "\u2078", "8", "\u2079", "9")
	countedPlurals = set("years", "miles", "feet", "yards", "leagues", "acres", "pounds", "dollars", "tons", "men",
		"people", "persons", "soldiers", "troops", "horses", "ships", "inhabitants", "copies", "pages", "words", "days",
		"members", "sheep", "cattle", "head", "guns", "prisoners", "volumes", "hours")
)

// ordinalSuffix matches in any case, since book text sets dates in capitals ("FEBRUARY 6", "18TH"), and also matches
// "d", since old books write "2d" and "3d" for second and third.
const ordinalSuffix = `(?:[sS][tT]|[nN][dD]|[rR][dD]|[tT][hH]|d)`

// numberShape bounds its digit runs so a match attempt on a long run stays short; a longer run is read in pieces.
const numberShape = `\d{1,30}(?:,\d{3}){0,10}(?:\.\d{1,30})?`

var (
	superscriptPattern = compile(`([A-Za-z0-9]{1,30})([\x{2070}\x{b9}\x{b2}\x{b3}\x{2074}-\x{2079}]{1,5})`)
	// A round number before "s." is a decade ("in his 30s.") unless money is near ("his wages were 40s. a week").
	// Indian grouping ("1,00,000", one lakh) is read as the plain number.
	indianGroupingPattern = compile(`\b\d{1,2}(?:,\d\d){1,5},\d{3}\b`)
	// A year before an era is read as a year ("843 B.C." as "eight forty three B.C.").
	eraYearPattern    = compile(`\b(\d{1,4}) ?(B\.\s?C\.(?:\s?E\.)?|A\.\s?D\.|C\.\s?E\.|BCE|BC|AD|CE)`)
	ordinalPattern    = compile(`\b(\d{1,30}(?:,\d{3}){0,10})(` + ordinalSuffix + `)\b`)
	decadePattern     = compile(`'?\b(\d{0,3}0)'?s\b`)
	yearNumberPattern = compile(`\b(1\d{3}|20\d{2})\b`)
	negativePattern   = compile(`(^|[\s(:])-(\d)`)
	cardinalPattern   = compile(numberShape)
	// Book text sometimes sets a number against a word or punctuation ("laurel,44", "(29)And"); the number read as
	// words needs a space there.
	spaceBeforeNumberPattern = compile(`([A-Za-z;!?)\]]|[A-Za-z],|[A-Za-z?!.,][\"']+)(\d)`)
	spaceAfterNumberPattern  = compile(`(\d[)\]])([A-Za-z])`)
	yearAfterPattern         = compile(`^(?:[.,]\d|%)`)
	fourDigitsPattern        = compile(`^\d{4}$`)
	yearWordBeforePattern    = compile(`\b[Yy]ear $`)
)

// number is the value of a run of ASCII digits, saturated far above any value a rule compares it with.
func number(digits string) int {
	digits = strings.TrimLeft(digits, "0")
	if len(digits) > 18 {
		return math.MaxInt>>1 + 1
	}
	value, _ := strconv.Atoi(digits)
	return value
}

// compareNumbers orders two runs of digits by value, whatever their length.
func compareNumbers(a, b string) int {
	a, b = strings.TrimLeft(a, "0"), strings.TrimLeft(b, "0")
	if len(a) != len(b) {
		return len(a) - len(b)
	}
	return strings.Compare(a, b)
}

// noun is a word's singular and plural.
type noun struct{ singular, plural string }

// counted is the singular after a written count of one and the plural after any other.
func (n noun) counted(written string) string {
	if written == "1" {
		return n.singular
	}
	return n.plural
}

// twoDigitWords reads two digits as a time's minutes or a year's last two digits: "05" as "oh five", "30" as "thirty".
func twoDigitWords(digits string) string {
	if value := number(digits); value < 10 {
		return "oh " + digitNames[value]
	}
	return smallNumberWords(number(digits))
}

func digitWords(digits string) string {
	return joined(strings.Split(digits, ""), " ", func(digit string) string { return digitNames[digit[0]-'0'] })
}

func smallNumberWords(value int) string {
	if value < 20 {
		return smallNumberNames[value]
	}

	words := tensNames[value/10]
	if value%10 != 0 {
		words += " " + smallNumberNames[value%10]
	}
	return words
}

// hundredsWords reads a number below one thousand with "and" after the hundreds ("one hundred and eighty").
func hundredsWords(value int) string {
	hundreds, rest := value/100, value%100
	if hundreds == 0 {
		return smallNumberWords(rest)
	}

	words := smallNumberNames[hundreds] + " hundred"
	if rest != 0 {
		words += " and " + smallNumberWords(rest)
	}
	return words
}

func integerWords(value int) string {
	return integerTextWords(strconv.Itoa(value))
}

type thousandsGroup struct {
	value, scale int
}

// integerTextWords reads a whole number of any length: 1001 as "one thousand and one", and one past the scale names
// digit by digit.
func integerTextWords(digits string) string {
	digits = strings.TrimLeft(digits, "0")
	if digits == "" {
		return "zero"
	}

	if len(digits) > 3*len(scaleNames) {
		return digitWords(digits)
	}

	value, _ := strconv.Atoi(digits)
	var groups []thousandsGroup
	for scale := 0; value > 0; scale++ {
		if part := value % 1000; part != 0 {
			groups = append([]thousandsGroup{{part, scale}}, groups...)
		}
		value /= 1000
	}

	words := make([]string, 0, len(groups)+1)
	for _, g := range groups {
		if g.scale == 0 && len(groups) > 1 && g.value < 100 {
			words = append(words, "and")
		}

		spoken := hundredsWords(g.value)
		if g.scale != 0 {
			spoken += " " + scaleNames[g.scale]
		}
		words = append(words, spoken)
	}
	return strings.Join(words, " ")
}

// numberWords reads a written number ("1,250", "-3", "2.5"); the digits after a decimal point are read one by one.
func numberWords(written string) string {
	sign := ""
	if strings.HasPrefix(written, "-") {
		sign = "minus "
	}

	whole, decimals, _ := strings.Cut(strings.ReplaceAll(strings.TrimLeft(written, "-"), ",", ""), ".")
	words := integerTextWords(whole)

	if decimals != "" {
		words += " point " + digitWords(decimals)
	}
	return sign + words
}

// yearWords reads a year: 1999 as "nineteen ninety nine", 1905 as "nineteen oh five", 2005 as "two thousand five".
func yearWords(year int) string {
	century, rest := year/100, year%100
	switch {
	case year%1000 == 0 || (2000 < year && year < 2010):
		return strings.ReplaceAll(integerWords(year), " and ", " ")
	case rest == 0:
		return smallNumberWords(century) + " hundred"
	case rest < 10:
		return smallNumberWords(century) + " oh " + digitNames[rest]
	}
	return smallNumberWords(century) + " " + smallNumberWords(rest)
}

// ordinalWords reads cardinal words as an ordinal: "twenty one" as "twenty first".
func ordinalWords(words string) string {
	cut := strings.LastIndex(words, " ") + 1
	head, last := words[:cut], words[cut:]
	switch {
	case ordinalNames[last] != "":
		last = ordinalNames[last]
	case strings.HasSuffix(last, "y"):
		last = last[:len(last)-1] + "ieth"
	default:
		last += "th"
	}
	return head + last
}

// pluralWords reads number words in the plural, as a decade is said: "nineteen ninety" as "nineteen nineties".
func pluralWords(words string) string {
	if strings.HasSuffix(words, "y") {
		return words[:len(words)-1] + "ies"
	}
	return words + "s"
}

func spelled(letters string) string {
	return strings.Join(strings.Split(strings.ToUpper(letters), ""), " ")
}

func ordinalSuffixOf(value int) string {
	if 11 <= value%100 && value%100 <= 13 {
		return "th"
	}

	switch value % 10 {
	case 1:
		return "st"
	case 2:
		return "nd"
	case 3:
		return "rd"
	}
	return "th"
}

// houseNumberWords reads a house number in pairs, as addresses are said: "123" as "one twenty three", "1600" as
// "sixteen hundred", "12345" digit by digit.
func houseNumberWords(digits string) string {
	switch len(digits) {
	case 3, 4:
		lead, rest := digits[:len(digits)-2], digits[len(digits)-2:]
		restWords := "hundred"
		if rest != "00" {
			restWords = twoDigitWords(rest)
		}
		return smallNumberWords(number(lead)) + " " + restWords
	case 5:
		return digitWords(digits)
	}
	return integerTextWords(digits)
}

// superscriptWords reads "x\u00b3" as "x cubed" and "10\u2076" as "10 to the sixth"; after a word it is a footnote
// mark ("walls\u00b9").
func superscriptWords(m match) string {
	base, exponent := m.group(1), m.group(2)
	if isAlpha(base) && len(base) > 1 {
		return base
	}
	return powerWords(base, superscriptDigits.Replace(exponent))
}

func isYear(value int) bool {
	return 1000 <= value && value <= 2099
}

// fractionWords reads "3/4" as "three quarters"; after a whole number one part is "a half" or "an eighth" ("2 1/2" as
// "2 and a half").
func fractionWords(numerator, denominator string, afterWhole bool) string {
	one := compareNumbers(numerator, "1") == 0
	ordinal := ordinalWords(integerTextWords(denominator))
	names, ok := fractionNames[strings.TrimLeft(denominator, "0")]
	if !ok {
		names = noun{ordinal, ordinal + "s"}
	}

	name := names.plural
	if one {
		name = names.singular
	}

	if afterWhole && one {
		return article(name) + " " + name
	}
	return integerTextWords(numerator) + " " + name
}

// eraYear reads "843 B.C." as "eight forty three B.C." and "1200 B.C." as "twelve hundred B.C.".
func eraYear(m match) (string, bool) {
	written, era := m.group(1), m.group(2)
	if isAlpha(string(nextCharacter(m))) {
		return "", false
	}

	switch len(written) {
	case 4:
		return yearWords(number(written)) + " " + era, true
	case 3:
		return houseNumberWords(written) + " " + era, true
	}
	return integerWords(number(written)) + " " + era, true
}

func ordinal(m match) (string, bool) {
	written, suffix := m.group(1), m.group(2)
	// Only books write "d" after a number, for an old ordinal ("2d") or pence ("1d"). Here it is left for the letter
	// it is ("3d printing", "7d").
	if suffix == "d" {
		return "", false
	}
	return ordinalWords(numberWords(written)), true
}

// decade reads "1990s" as "nineteen nineties" and "'80s" as "eighties".
func decade(m match) string {
	value := number(m.group(1))
	if isYear(value) {
		return pluralWords(yearWords(value))
	}
	return pluralWords(integerWords(value))
}

// yearNumber reads a four-digit number from 1000 to 2099 as a year, as a reader says it whether it is one or not
// ("1500 men" as "fifteen hundred men").
func yearNumber(m match) (string, bool) {
	previous := previousCharacter(m)
	following := textAfter(m)
	if strings.ContainsRune(moneySigns+".,-/:", previous) || unicode.IsDigit(previous) ||
		yearAfterPattern.MatchString(following) {
		return "", false
	}

	// Before a counted plural it is a count ("1500 years" as "one thousand five hundred years").
	if countedPlurals[lowerWordAfterPattern.firstGroup(following)] {
		return "", false
	}
	return yearWords(number(m.group(1))), true
}

// cardinal reads any other number; five or more digits after a word naming a reference ("Order 7741890") are read
// digit by digit, as is a long run written without commas and a number written with a leading zero; "911" is always
// the phone number.
func cardinal(m match) string {
	written := m.group(0)
	whole, decimals, decimal := strings.Cut(written, ".")
	digits := strings.ReplaceAll(whole, ",", "")
	preceding := textBefore(m)
	commas := strings.Contains(written, ",")
	reference := !commas && len(digits) >= 5 && referenceBeforePattern.MatchString(preceding)
	barcode := !commas && !decimal && longDigitRun(digits)
	var words string
	switch {
	case written == "911" || (len(digits) > 1 && digits[0] == '0') || reference || barcode:
		words = digitWords(digits)
		if decimal {
			words += " point " + digitWords(decimals)
		}
	case fourDigitsPattern.MatchString(written) && yearWordBeforePattern.MatchString(preceding):
		words = yearWords(number(written))
	default:
		words = numberWords(written)
	}

	if isAlphanumeric(nextCharacter(m)) {
		words += " "
	}
	return words
}

// pairedNumber reads "Room 412" as "Room four twelve", "Suite 1005" as "Suite ten oh five" and "rooms 310 to 318" as
// "rooms three ten to three eighteen".
func pairedNumber(m match) (string, bool) {
	if continuesNumber(m) {
		return "", false
	}

	words := m.group(1) + houseNumberWords(m.group(2))
	if last := m.group(3); last != "" {
		words += " to " + houseNumberWords(last)
	}
	return words, true
}

func article(word string) string {
	if strings.ContainsRune("aeiou", rune(word[0])) {
		return "an"
	}
	return "a"
}
