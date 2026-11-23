package text

import (
	"slices"
	"strings"
)

var (
	dottedPhonePattern        = compile(`\b(\d{3})\.(\d{3})\.(\d{4})\b`)
	internationalPhonePattern = compile(`\+(\d{1,3})((?:[- ]\d{1,4}){2,5})\b`)
	nanpPhonePattern          = compile(`(\+1 )?\((\d{3})\) ?(\d{3})-(\d{4})\b`)
	digitGroupsPattern        = compile(
		`(\+?)\b([A-Z0-9]{0,9}[0-9][A-Z0-9]{0,9}(?: [A-Z0-9]{0,9}[0-9][A-Z0-9]{0,9}){1,10})\b`)
	phoneSplitPattern = compile(`[- ]`)
)

// internationalPhone reads "+44-20-7946-0958" as "plus four four, two zero, seven nine four six, zero nine five
// eight".
func internationalPhone(m match) string {
	groups := append([]string{m.group(1)}, phoneSplitPattern.Split(strings.Trim(m.group(2), "- "), -1)...)
	return "plus " + phoneWords(groups)
}

func dottedPhone(m match) string {
	return phoneWords([]string{m.group(1), m.group(2), m.group(3)})
}

func nanpPhone(m match) string {
	country, area, exchange, line := m.group(1), m.group(2), m.group(3), m.group(4)
	spoken := phoneWords([]string{area, exchange, line})
	if country != "" {
		return "plus one, " + spoken
	}
	return spoken
}

// digitGroups reads digit groups set apart by spaces ("020 7946 0018") digit by digit, a pause between groups.
func digitGroups(m match) (string, bool) {
	plus, groups := m.group(1), m.group(2)
	groupList := strings.Split(groups, " ")
	anyYear := slices.ContainsFunc(groupList, isYearGroup)
	allYears := !slices.ContainsFunc(groupList, func(group string) bool { return !isYearGroup(group) })

	// A year next to a count is prose ("In 1850 12 men died"), not a reference.
	if continuesNumber(m) || asciiDigitCount(groups) < 6 || allYears || (len(groupList) == 2 && anyYear) {
		return "", false
	}

	// Short groups of one length with nothing naming a phone number or reference are counts ("12 14 16").
	if sameShortGroups(groupList) && plus == "" && !referenceBeforePattern.MatchString(textBefore(m)) {
		return "", false
	}

	spoken := spokenGroupRuns(groupList)
	if allDigits(groupList) {
		spoken = phoneWords(groupList)
	}

	if plus != "" {
		return "plus " + spoken, true
	}
	return spoken, true
}

func asciiDigitCount(text string) int {
	digits := 0
	for _, character := range []byte(text) {
		if isASCIIDigit(character) {
			digits++
		}
	}
	return digits
}

func isYearGroup(group string) bool {
	return isDigits(group) && len(group) == 4 && 1500 <= number(group) && number(group) <= 2099
}

// sameShortGroups is whether the groups are digits of one length up to three, none with a leading zero.
func sameShortGroups(groupList []string) bool {
	return len(groupList[0]) <= 3 && !slices.ContainsFunc(groupList, func(group string) bool {
		return len(group) != len(groupList[0]) || !isDigits(group) || group[0] == '0'
	})
}

// spokenGroupRuns reads each group's digit runs digit by digit and its capital runs spelled, a pause between groups.
func spokenGroupRuns(groupList []string) string {
	return joined(groupList, ", ", func(group string) string { return identifierPartWords(group, false, true) })
}

// phoneWords reads a phone number's digit groups, each digit by digit with a pause between groups. A group of one
// digit and two zeros is said in hundreds, as people say a toll-free "800" ("eight hundred").
func phoneWords(groups []string) string {
	return joined(groups, ", ", func(group string) string {
		if len(group) == 3 && group[0] != '0' && group[1:] == "00" {
			return digitNames[group[0]-'0'] + " hundred"
		}
		return digitWords(group)
	})
}

var (
	// After "call" a short number is a count ("Call 125 men"), except the short service numbers.
	shortPhoneNumbers   = set("000", "101", "111", "112", "911", "999")
	keywordAfterPattern = compile(`^(?:[.,:/-]\d|[-/])`)
)

// keywordDigits reads a phone number or PIN after "call", "PIN" or "code" ("Call 111", "enter 1234") digit by digit.
func keywordDigits(m match) (string, bool) {
	keyword, numbers := m.group(1), m.group(2)
	if keywordAfterPattern.MatchString(textAfter(m)) {
		return "", false
	}

	numberList := strings.Split(numbers, " or ")
	// A round number is a count ("Call 300 people"), not a phone number.
	for _, written := range numberList {
		if strings.HasSuffix(written, "00") && written[0] != '0' {
			return "", false
		}
	}

	shortest := 3
	lower := strings.ToLower(keyword)
	switch {
	case strings.HasPrefix(lower, "call") || strings.HasPrefix(lower, "dial") || strings.HasPrefix(lower, "ring") ||
		strings.HasPrefix(lower, "text"):
		shortest = 5
	case strings.HasPrefix(lower, "enter"):
		shortest = 4
	}

	spoken := make([]string, 0, len(numberList))
	for _, written := range numberList {
		if len(written) < shortest && written[0] != '0' && !shortPhoneNumbers[written] {
			return "", false
		}
		spoken = append(spoken, digitWords(written))
	}

	if lower == "ext. " {
		keyword = "extension "
	}
	return keyword + strings.Join(spoken, " or "), true
}

var (
	// A number after one of these keywords is a phone number or PIN read digit by digit ("Call 111", "enter 1234").
	keywordDigitsPattern = compile(`\b((?i:call|dial|ring|text|extension|ext\.|enter|ends in|pin(?: code)?(?: is)?` +
		`|code(?: is)?) )(\d{3,}(?: or \d{3,})*)\b`)
)
