package text

import (
	"slices"
	"strconv"
	"strings"
)

var (
	// The period of "5 p.m." is the abbreviation's and may also end the sentence, and "5 p.m!" has a broken one.
	// Written as "PM" first, the period stays only where a sentence ends: at the end of the text, or before a capital
	// past spaces and an opening quote, unless the capital starts a time zone ("5 p.m. ET").
	meridiemEndPattern      = compile(`(\d\s?)([aApP])\.\s?[mM]\.`)
	meridiemPattern         = compile(`(\d\s?)([aApP])\.\s?[mM]\.?`)
	meridiemEndAfterPattern = compile(`^\s*$|^\s+["']?[A-Z]`)
	timeZoneAfterPattern    = compile(`^\s+["']?(?:UTC|[A-Z]{1,3}T)\b`)
	// "at 8.45" and "the 10.15 train" are British times; a dotted number elsewhere stays a decimal.
	dottedTimePattern = compile(`\b((?i:at|by|until|till|before|after|around|about|from|the) )(\d{1,2})\.([0-5]\d)\b`)
	timeAfterPattern  = compile(`^(?:\s*(?:[.,;:!?)]|$)| (?:AM|PM|o'clock|in the|on|and|or|to|until|till|when|that` +
		`|this|tonight|tomorrow|today|sharp|train|bus|boat|ferry|flight|express|service|coach|tram|mail)\b)`)
	notTimeAfterPattern    = compile(`^ ?(?:%|(?i:percent|per|times|points?)\b)`)
	transportPattern       = compile(`^ (?:train|bus|boat|ferry|flight|express|service|coach|tram|mail)\b`)
	dottedTimeRangePattern = compile(`\b(\d{1,2})\.(\d{2})( ?- ?| to | until | till )(\d{1,2})\.(\d{2})\b`)
	timeRangePattern       = compile(`\b(\d{1,2}(?::\d{2})?(?: ?[AaPp][Mm])?)-(\d{1,2}(?::\d{2})?(?: ?[AaPp][Mm])?)\b`)
	// An hour past noon or of 00 makes a text use the 24-hour clock: a text keeps one clock convention throughout,
	// like its day/month order, so this looks at the whole text.
	twentyFourHourPattern = compile(`\b(?:00|1[3-9]|2[0-3]):[0-5]\d\b`)
	// "12:00 noon" and "12:00 midnight" are "twelve noon" and "twelve midnight".
	noonAfterPattern          = compile(`^ (?i:noon|midnight)\b`)
	clockTimePattern          = compile(`\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?: ?([AaPp][Mm])\b)?`)
	militaryTimePattern       = compile(`\b([01]\d|2[0-3])([0-5]\d) hours\b`)
	meridiemTimePattern       = compile(`\b(\d{1,2})(?:[.:](\d{2}))? ?([AaPp][Mm])\b`)
	betweenNearPattern        = compile(`\b[Bb]etween\b`)
	pairedHoursAfterPattern   = compile(`^ ?(?:hours\b|(?:to|until|till|-) ?\d)`)
	pairedHoursBeforePattern  = compile(`\d ?(?:to|until|till|-) ?$`)
	betweenHoursAfterPattern  = compile(`^ ?(?:hours\b|(?:to|until|till|-|and|or) ?\d)`)
	betweenHoursBeforePattern = compile(`\d ?(?:to|until|till|-|and|or) ?$`)
)

// meridiemLetters writes "5 p.m." as "5 PM", with a period after it only where it also ends a sentence.
func meridiemLetters(text string) string {
	text = meridiemEndPattern.replaceWhere(text, func(m match) (string, bool) {
		rest := m.text[m.end():]
		if !meridiemEndAfterPattern.MatchString(rest) || timeZoneAfterPattern.MatchString(rest) {
			return "", false
		}
		return m.group(1) + strings.ToUpper(m.group(2)) + "M.", true
	})
	return meridiemPattern.replaceFunc(text, func(m match) string {
		return m.group(1) + strings.ToUpper(m.group(2)) + "M"
	})
}

// dottedTimeRange reads "9.30 to 11.30" as "9:30 to 11:30", times instead of decimals.
func dottedTimeRange(m match) (string, bool) {
	startHour, startMinute, joiner, endHour, endMinute := m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
	hoursValid := !slices.ContainsFunc([]string{startHour, endHour}, func(hour string) bool {
		return hour[0] == '0' && len(hour) != 2
	})
	// Times in a range fall on five-minute marks; decimals ("from 1.18 to 1.21") mostly do not.
	minutesValid := !slices.ContainsFunc([]string{startMinute, endMinute}, func(minute string) bool {
		return number(minute)%5 != 0 || number(minute) > 55
	})

	if continuesNumber(m) || !hoursValid || !minutesValid || max(number(startHour), number(endHour)) > 23 {
		return "", false
	}

	if strings.TrimSpace(joiner) == "-" {
		joiner = " to "
	}
	return strconv.Itoa(number(startHour)) + ":" + startMinute + joiner + strconv.Itoa(number(endHour)) + ":" +
		endMinute, true
}

// timeRange reads "14:00-16:00" or "9am-5pm" with "to" for the hyphen.
func timeRange(m match) (string, bool) {
	start, end := m.group(1), m.group(2)
	timed := slices.ContainsFunc([]string{start, end}, func(side string) bool {
		return strings.Contains(side, ":") || strings.HasSuffix(side, "m") || strings.HasSuffix(side, "M")
	})

	if !timed || continuesNumber(m) {
		return "", false
	}
	return start + " to " + end, true
}

// saysHours is whether a time written on the 24-hour clock ("19:00", "08:00", "00:30") says "hours" after it; a
// time in a range does not ("14:00-16:00", "between 09:00 and 17:00"), nor one already followed by "hours".
func saysHours(m match, hour string) bool {
	military := number(hour) >= 13 || (len(hour) == 2 && hour[0] == '0')
	preceding := textBefore(m)
	afterPattern, beforePattern := pairedHoursAfterPattern, pairedHoursBeforePattern
	if betweenNearPattern.MatchString(before(preceding, len(preceding), 30)) {
		afterPattern, beforePattern = betweenHoursAfterPattern, betweenHoursBeforePattern
	}

	paired := afterPattern.MatchString(textAfter(m)) || beforePattern.MatchString(preceding)
	return military && !paired
}

// clockTime reads "14:30" as "fourteen thirty", "07:05" as "seven oh five" and "2:03:45" in hours, minutes and
// seconds; a time on the hour is "o'clock", or on the 24-hour clock "hundred hours", and "12:00 noon" is "twelve
// noon".
func clockTime(m match, twentyFourHour bool) (string, bool) {
	hour, minute, second, meridiem := m.group(1), m.group(2), m.group(3), strings.ToUpper(m.group(4))
	if continuesNumber(m) || number(hour) > 23 || number(minute) > 59 {
		return "", false
	}

	hourWords := clockHourWords(hour, twentyFourHour)
	if second != "" {
		hourName := hourUnit.counted(strconv.Itoa(number(hour)))
		minuteName := minuteUnit.counted(strconv.Itoa(number(minute)))
		secondName := secondUnit.counted(strconv.Itoa(number(second)))
		return hourWords + " " + hourName + " " + integerWords(number(minute)) + " " + minuteName + " and " +
			integerWords(number(second)) + " " + secondName, true
	}

	following := textAfter(m)
	hours := ""
	if saysHours(m, hour) {
		hours = " hours"
	}

	if minute == "00" {
		switch {
		case meridiem != "":
			return hourWords + " " + meridiem, true
		case noonAfterPattern.MatchString(following):
			if hour == "00" || hour == "12" {
				return "twelve", true
			}
			return hourWords, true
		case twentyFourHour:
			return hourWords + " hundred" + hours, true
		}
		return hourWords + " o'clock", true
	}

	if twentyFourHour && hour == "00" {
		return hourWords + " " + twoDigitWords(minute) + hours, true
	}

	if meridiem != "" {
		return hourWords + " " + twoDigitWords(minute) + " " + meridiem, true
	}
	return hourWords + " " + twoDigitWords(minute), true
}

func clockHourWords(hour string, twentyFourHour bool) string {
	hourWords := integerWords(number(hour))
	// "08:00 and 20:00" must not run together as "eight hundred and twenty hundred".
	if twentyFourHour && len(hour) == 2 && hour[0] == '0' && hour != "00" {
		hourWords = "oh " + hourWords
	}
	return hourWords
}

// militaryTime reads "0800 hours" as "oh eight hundred hours".
func militaryTime(m match) string {
	hour, minute := m.group(1), m.group(2)
	hourWords := clockHourWords(hour, true)
	if minute == "00" {
		return hourWords + " hundred hours"
	}
	return hourWords + " " + twoDigitWords(minute) + " hours"
}

// meridiemTime reads "8am" or "7.30 pm" as "eight AM" or "seven thirty PM".
func meridiemTime(m match) (string, bool) {
	hour, minute, meridiem := m.group(1), m.group(2), m.group(3)
	if number(hour) > 12 || (minute != "" && number(minute) > 59) || strings.ContainsRune(".:", previousCharacter(m)) {
		return "", false
	}

	spokenMinute := ""
	if minute != "" && minute != "00" {
		spokenMinute = " " + twoDigitWords(minute)
	}
	return integerWords(number(hour)) + spokenMinute + " " + strings.ToUpper(meridiem), true
}

// dottedTime reads "at 8.45" as "at 8:45"; "the 10.15" only before a train or a bus.
func dottedTime(m match) (string, bool) {
	word, hour, minute := m.group(1), m.group(2), m.group(3)
	following := textAfter(m)
	// A time falls on a five-minute mark; "from 1.18 to 1.21" is decimals. After "at", "by", "until" or "till" any
	// word may follow ("At 8.45 we left"); after the others the next word must say time.
	if continuesAfter(m) || number(hour) > 23 || number(minute)%5 != 0 || notTimeAfterPattern.MatchString(following) {
		return "", false
	}

	switch strings.ToLower(strings.TrimSpace(word)) {
	case "at", "by", "until", "till":
	default:
		if !timeAfterPattern.MatchString(following) {
			return "", false
		}
	}

	if strings.EqualFold(word, "the ") && !transportPattern.MatchString(following) {
		return "", false
	}
	return word + strconv.Itoa(number(hour)) + ":" + minute, true
}

var (
	// Seconds written short after a word that sets a duration ("timeout to 30s", "every 5s") and before the end of
	// the phrase; "the 30s" and "her 30s" are decades.
	shortSecondsPattern = compile(`\b([A-Za-z]+) (\d{1,3}(?:\.\d+)?)s\b`)
	secondsBefore       = set("to", "every", "after", "wait", "waits", "waited", "within", "timeout", "takes", "took",
		"for", "in", "than", "about", "only", "just", "max", "maximum", "minimum", "least")
	secondsNextPattern = compile(`^(?:[.,;:!?)]|$| (?:and|or|then|before|after|later|left|ago|each|per|of|to|timeout` +
		`|delay|max|at|with|but|if|while|until|when|between|on|for|so)\b)`)
)

// shortSeconds reads "timeout to 30s" as "timeout to 30 seconds"; without a duration word on both sides it stays a
// decade or a plural ("in 90s fashion", "two 5s").
func shortSeconds(m match) (string, bool) {
	word, written := m.group(1), m.group(2)
	if !secondsBefore[strings.ToLower(word)] || !secondsNextPattern.MatchString(textAfter(m)) {
		return "", false
	}
	return word + " " + written + " " + secondUnit.counted(written), true
}
