package text

import (
	"strings"
)

var (
	// A month abbreviation without a day ("in Feb.") after a word that sets a time.
	monthAlonePattern = compile(`\b((?i:in|by|due|of|until|till|since|from|before|after|last|next|early|late|mid` +
		`|during|this|every) )(` + monthAbbreviationNames + `)\.`)
	// "of" sets a time only after a day ("the 5th of Mar."); "Lord of Mar." is a place.
	dayBeforeOfPattern = compile(`(?i:\d(?:st|nd|rd|th|d)?|\b(?:first|second|third|fourth|fifth|sixth|seventh` +
		`|eighth|ninth|tenth|eleventh|twelfth|[a-z]*teenth|twentieth|thirtieth|end|middle|beginning|ides)) $`)
	// A month abbreviation before a year reads as the month ("In Aug. 1932").
	monthYearPattern = compile(`\b(` + monthAbbreviationNames + `)\. (\d{4})\b`)
	days             = dayAbbreviationTable()
	dayNames         = longestFirst(sortedKeys(days))
	dayPattern       = compile(`\b(` + dayNames + `)(\.?)(?:-(` + dayNames + `)(\.?))?`)
	// Words that set a time before a lone day ("until Fri.", "The Tues. meeting"); "the northern extremity of Mon."
	// is Anglesey (Welsh Mon).
	dayBeforePattern = compile(`\b(?i:on|until|till|by|from|to|since|next|last|this|every|each|before|after|the|a) $`)
	// Two different abbreviated months in one text make every abbreviated month a month ("the Jan. and Feb. numbers").
	monthAbbreviationPattern = compile(`\b(` + monthAbbreviationNames + `)\.`)

	spaceDigitPattern       = compile(`^ \d`)
	dayPeriodPattern        = compile(`\b(` + dayNames + `)\.`)
	weekdayPattern          = compile(`\b(?:Mon|Tue|Wed|Thu|Fri)`)
	monthAfterDayPattern    = compile(`^, ([A-Za-z]{3,9})\.? \d`)
	dayOfMonthAfterPattern  = compile(`^,? (?:the )?\d{1,2}(?:st|nd|rd|th)\b`)
	capitalizedAfterPattern = compile(`^ ([A-Z][a-z]+)\.? ?(\d?)`)
)

// monthAlone reads "in Feb." as "in February", the period kept only where it also ends the sentence.
func monthAlone(m match) (string, bool) {
	word, month := m.group(1), m.group(2)
	if spaceDigitPattern.MatchString(textAfter(m)) {
		return "", false
	}

	lower := strings.ToLower(word)
	if lower == "of " && !dayBeforeOfPattern.MatchString(textBefore(m)) {
		return "", false
	}

	// "before Jan." at a sentence end may name a person.
	switch lower {
	case "before ", "after ", "by ", "from ":
		if sentenceEnd(m.text, m.end()) {
			return "", false
		}
	}
	return word + monthAbbreviations[month] + sentencePeriod(m.text, m.end()), true
}

// monthInList reads an abbreviated month in a text that abbreviates two or more months ("the Jan. and Feb. numbers").
func monthInList(m match) string {
	return monthAbbreviations[m.group(1)] + sentencePeriod(m.text, m.end())
}

// monthList is whether text abbreviates two or more different months, which makes each abbreviation a month.
func monthList(text string) bool {
	return monthAbbreviationPattern.distinctGroups(text) >= 2
}

// dayList is whether text abbreviates two or more different days, one a weekday, which makes "Sat." and "Sun." days.
func dayList(text string) bool {
	return dayPeriodPattern.distinctGroups(text) >= 2 && weekdayPattern.MatchString(text)
}

// day reads "Mon.-Fri." as "Monday to Friday", "Fri." as "Friday" and "Fri, Oct 10" as "Friday, Oct 10"; "Sat." and
// "Sun." alone are too often words, so they are read only in a range, beside other abbreviated days or before a date.
func day(m match, daysListed bool) (string, bool) {
	first, firstPeriod, last := m.group(1), m.group(2), m.group(3)
	if isAlpha(string(nextCharacter(m))) {
		return "", false
	}

	if last != "" {
		return weekdayRange(m), true
	}

	following := textAfter(m)
	if monthDateAfter(following) {
		return days[first], true
	}

	// A day of the month after it is a date ("Sat the 8th", "Sun 12th").
	if dayOfMonthAfterPattern.MatchString(following) {
		return days[first] + keptPeriod(m, firstPeriod), true
	}

	if firstPeriod == "" || ((first == "Sat" || first == "Sun") && !daysListed) {
		return "", false
	}

	// "Mon. May came" has no date after the month.
	nextWord, named := capitalizedAfterPattern.find(following)
	if named && monthName(nextWord.group(1)) != "" && nextWord.group(2) != "" {
		return days[first], true
	}

	if loneDayAsWritten(spotOf(m), daysListed) {
		return "", false
	}
	return days[first] + sentencePeriod(m.text, m.end()), true
}

// weekdayRange reads "Mon.-Fri." as "Monday to Friday".
func weekdayRange(m match) string {
	return days[m.group(1)] + " to " + days[m.group(3)] + keptPeriod(m, m.group(4))
}

// monthDateAfter is whether a month and its day follow a day ("Fri, Oct 10").
func monthDateAfter(following string) bool {
	return monthName(monthAfterDayPattern.firstGroup(following)) != ""
}

// loneDayAsWritten is whether an abbreviated day stays as written: alone, with no word that sets a time before it and
// no date after it.
func loneDayAsWritten(s spot, daysListed bool) bool {
	lone := !daysListed && !spaceDigitPattern.MatchString(s.following)
	return lone && !dayBeforePattern.MatchString(s.preceding)
}

// dayAbbreviationTable maps each abbreviated day to its name: its first three letters, and "Tues", "Thur", "Thurs".
func dayAbbreviationTable() map[string]string {
	abbreviations := map[string]string{}
	for _, name := range weekdayNames {
		abbreviations[name[:3]] = name
	}

	for _, longer := range []string{"Tues", "Thur", "Thurs"} {
		abbreviations[longer] = abbreviations[longer[:3]]
	}
	return abbreviations
}
