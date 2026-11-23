package text

import (
	"strconv"
	"strings"
)

var (
	romanNumeralPattern = compile(`\b([A-Za-z]+) ([IVXLCDM]+|[ivxlcdm]+)\b(\.?)`)
	// Before these an uppercase numeral is an ordinal ("the XV century" as "the fifteenth century").
	romanCenturyPattern = compile(
		`\b([IVXL]{1,7}) ((?i:century|centuries|dynasty|legion|army|corps|congress|olympiad))\b`)
	// Before these a numeral is a count ("Chapter IV", "World War II").
	sectionWords = set("chapter", "part", "book", "volume", "act", "scene", "section", "article", "phase", "war",
		"psalm", "canto", "stave", "plate", "page", "appendix", "lecture", "letter", "sermon", "title", "bowl")
	// Words that also spell a Roman numeral, left as written ("the bowl mix well", "Canto di S. Maria", "book liv").
	numeralLookalikes = set("mix", "di", "dix", "mi", "dim", "mid", "vim", "lid", "liv", "mild", "civil", "vivid",
		"dill")
	// After these a numeral is a regnal number ("Henry VIII"); after any other name it is left alone ("Ward II").
	regnalNames = set("Henry", "Edward", "George", "William", "Richard", "Charles", "James", "John", "Louis",
		"Napoleon", "Elizabeth", "Mary", "Anne", "Victoria", "Philip", "Peter", "Paul", "Pius", "Leo", "Gregory",
		"Benedict", "Innocent", "Clement", "Urban", "Boniface", "Alexander", "Frederick", "Ferdinand", "Francis",
		"Joseph", "Leopold", "Otto", "Ivan", "Nicholas", "Catherine", "Constantine", "Ptolemy", "Ramesses", "Harold",
		"David", "Robert", "Stephen", "Isabella", "Gustav", "Gustavus", "Christian", "Olaf", "Haakon", "Henri",
		"Wilhelm", "Friedrich", "Karl", "Rudolf", "Pedro", "Juan", "Carlos", "Felipe", "Umberto", "Emmanuel", "Selim",
		"Suleiman", "Mehmed", "Murad", "Darius", "Xerxes", "Antiochus", "Cleopatra", "Thutmose", "Amenhotep", "Sixtus",
		"Julius", "Adrian", "Hadrian", "Martin", "Eugene", "Sylvester", "Augustus", "Edmund", "Ethelred", "Alfred",
		"Kenneth", "Duncan")
	// Capitalized words after "Henry V." that start a new sentence instead of continuing a name ("Henry V. Now he
	// ruled").
	regnalSentenceWords = union(sentenceStarters, regnalNames, set("Now", "Here", "Thus", "Soon", "Later",
		"Meanwhile", "Yet", "Still", "Hence", "England", "France", "Spain", "Scotland", "Paris", "London", "Rome",
		"Madrid", "Naples", "Vienna"))
	// A lone "I" is a numeral only before punctuation, "and", "of" or the end ("World War I", "Charles I, King of
	// England"); before any other word it is the pronoun ("In Chapter I was told").
	loneNumeralAfterPattern = compile(`^(?:\s*$|[.:,;!?)]| and\b| of\b|\s*--)`)
	surnameAfterPattern     = compile(`^\.\s+([A-Z][a-z]*)(\.?)`)
)

// sectionNumeral reads "Chapter IV" as "Chapter 4", or declines where the numeral is a letter or a pronoun.
func sectionNumeral(s spot, value int, period string) (string, bool) {
	word, numeral, afterNumeral := s.group(1), s.group(2), s.group(3)+s.following
	// "Appendix C" is a letter, and "the letter X" too; only I, V and X stand alone, after a capitalized word.
	if len(numeral) == 1 && (!strings.ContainsAny(strings.ToUpper(numeral), "IVX") || isLower(word)) {
		return "", false
	}

	if (numeral == "I" || numeral == "i") && !loneNumeralAfterPattern.MatchString(afterNumeral) {
		return "", false
	}
	return word + " " + strconv.Itoa(value) + period, true
}

// regnalNumeral reads "Henry VIII" or "Henry the VIII" as "Henry the 8th", or declines where the letter is an initial
// or a pronoun.
func regnalNumeral(s spot, value int, period string) (string, bool) {
	word, numeral, afterNumeral := s.group(1), s.group(2), s.group(3)+s.following
	if word == "the" {
		name, ok := capitalizedBeforePattern.find(s.preceding)
		if !ok || !regnalNames[name.group(1)] {
			return "", false
		}
	} else if !regnalNames[word] {
		return "", false
	}

	if numeral == "I" && !loneNumeralAfterPattern.MatchString(afterNumeral) {
		return "", false
	}

	if len(numeral) == 1 && letterIsInitial(afterNumeral) {
		return "", false
	}

	the := " the"
	if word == "the" {
		the = ""
	}
	return word + the + " " + strconv.Itoa(value) + ordinalSuffixOf(value) + period, true
}

// letterIsInitial is whether a one-letter numeral after a name is a person's initial: one before another name ("Mary
// V. Smith"), unless that word starts a sentence ("Henry V. In that year").
func letterIsInitial(afterNumeral string) bool {
	surname, ok := surnameAfterPattern.find(afterNumeral)
	return ok && (!regnalSentenceWords[surname.group(1)] || surname.group(2) != "")
}

// romanNumeral reads "Chapter IV" as "Chapter 4" and "Henry VIII" or "Henry the VIII" as "Henry the 8th".
func romanNumeral(m match) (string, bool) {
	s := spotOf(m)
	word, numeral, period := m.group(1), m.group(2), m.group(3)
	value, ok := romanValue(numeral)
	afterNumeral := period + s.following
	// "XIV-" and "XIV's" join a word; "IV.--THE" is a heading's dash.
	if !ok || numeralLookalikes[strings.ToLower(numeral)] || strings.HasPrefix(afterNumeral, "'") ||
		(strings.HasPrefix(afterNumeral, "-") && !strings.HasPrefix(afterNumeral, "--")) {
		return "", false
	}

	// The period after a numeral stays only where it also ends the sentence ("Louis XIV. was king").
	period = keptPeriod(m, period)

	if sectionWords[strings.ToLower(word)] {
		return sectionNumeral(s, value, period)
	}

	if isLower(numeral) || value >= 40 {
		return "", false
	}
	return regnalNumeral(s, value, period)
}

// romanCentury reads "the XV century" as "the fifteenth century"; "I" is an ordinal only after "the" ("the I Corps").
func romanCentury(m match) (string, bool) {
	numeral, noun := m.group(1), m.group(2)
	value, ok := romanValue(numeral)
	if !ok || (numeral == "I" && !strings.HasSuffix(m.text[:m.start()], "the ")) {
		return "", false
	}
	return ordinalWords(integerWords(value)) + " " + noun, true
}

var (
	romanValues = map[byte]int{'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
	romanDigits = []romanDigit{{1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"}, {100, "C"}, {90, "XC"}, {50, "L"},
		{40, "XL"}, {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"}}
)

type romanDigit struct {
	value   int
	letters string
}

// romanText writes value as a Roman numeral in its standard form.
func romanText(value int) string {
	var b strings.Builder
	for _, digit := range romanDigits {
		b.WriteString(strings.Repeat(digit.letters, value/digit.value))
		value %= digit.value
	}
	return b.String()
}

// romanValue is the value of a Roman numeral in either case written in its standard form, or false.
func romanValue(numeral string) (int, bool) {
	letters := strings.ToUpper(numeral)
	total := 0
	for i := range len(letters) {
		value := romanValues[letters[i]]
		following := 0
		if i+1 < len(letters) {
			following = romanValues[letters[i+1]]
		}

		if value < following {
			total -= value
		} else {
			total += value
		}
	}
	return total, 0 < total && total < 4000 && romanText(total) == letters
}
