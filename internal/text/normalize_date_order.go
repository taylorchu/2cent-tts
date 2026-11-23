package text

import (
	"strings"
	"time"
)

var (
	weekdayNames = []string{"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
	// A weekday next to a slashed date ("Thursday, 12/11") picks the reading that falls on it.
	weekdayDatePattern = compile(`\b(` + strings.Join(weekdayNames, "|") + `),? (\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b`)
	yearInTextPattern  = compile(`\b(1[89]\d\d|20\d\d)\b`)
	// A slashed day and month, with the date word before it that marks it as a date and not a fraction.
	slashedDatePattern = compile(`\b(?:((?i:on|by|until|till|from|to|arriving|arrives|due|shipped|delivered|before` +
		`|after|since|dated)) )?(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?\b`)
)

// calendarDate is the date of year, month and day, or false where no such day exists.
func calendarDate(year, month, day int) (time.Time, bool) {
	if year < 1 || !isMonth(month) || day < 1 {
		return time.Time{}, false
	}

	date := time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
	return date, date.Day() == day
}

// weekdayDayFirst is whether a weekday next to a slashed date ("Thursday, 12/11") matches only the day-first reading,
// in the date's year or the year the text names; the first weekday that matches only one reading decides.
func weekdayDayFirst(text string) bool {
	namedYear, hasYear := yearInTextPattern.find(text)
	for _, found := range weekdayDatePattern.findAll(text) {
		weekday, first, second, year := found.group(1), found.group(2), found.group(3), found.group(4)
		if year == "" && hasYear {
			year = namedYear.group(1)
		}

		yearValue := number(year)
		readings := [2]bool{}
		for i, order := range [2][2]int{{number(first), number(second)}, {number(second), number(first)}} {
			date, ok := calendarDate(yearValue, order[0], order[1])
			readings[i] = ok && weekdayNames[(int(date.Weekday())+6)%7] == weekday
		}

		if readings[0] != readings[1] {
			return readings[1]
		}
	}
	return false
}

// slashedDayFirst is whether the slashed dates of text are day first, decided once per text: by a date that reads only
// one way ("23/09" or "3/14"), a dotted date, or a weekday that matches one reading; month first without evidence.
func slashedDayFirst(text string) bool {
	monthFirst := false
	for _, found := range slashedDatePattern.findAll(text) {
		first, second := number(found.group(2)), number(found.group(3))
		if !isDayOfMonth(first) || !isDayOfMonth(second) {
			continue
		}

		if first > 12 && second <= 12 {
			return true
		}

		// Only a pair with a year or after a date word counts as a date from here on, not a fraction ("3/4").
		if (found.group(4) != "" || found.group(1) != "") && first <= 12 && 12 < second {
			monthFirst = true
		}
	}

	if dottedDatePattern.MatchString(text) {
		return true
	}

	if monthFirst {
		return false
	}

	return weekdayDayFirst(text)
}
