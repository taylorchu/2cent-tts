package text

import "strings"

var (
	// A house number, a street name and a street type ("123 N Main St NW"): the number is read in pairs, as addresses
	// are said ("one twenty three"), and the abbreviations in words.
	streetAddressPattern = compile(`\b(\d{2,5})([A-Z]?) ((?:[NS][EW]?|[EW])\.? )?((?:[A-Z][a-z]+|\d+(?:st|nd|rd|th))` +
		` (?:[A-Z][a-z]+ ){0,2})(` + longestFirst(append(sortedKeys(streets),
		sortedKeys(streetWords)...)) + `)\b(\.?)((?:,? (?:NE|NW|SE|SW|N|S|E|W)\b)?)`)
	states = map[string]string{
		"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
		"CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
		"IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
		"ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
		"MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
		"NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
		"ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
		"RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
		"UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
		"WI": "Wisconsin", "WY": "Wyoming", "DC": "D C", "PR": "Puerto Rico"}
	// State codes that are also words ("Hello, OK") are read as states only before a ZIP code.
	wordStates   = set("OK", "IN", "ME", "OR", "HI", "ID", "OH")
	statePattern = compile(`\b([A-Z][a-z]+(?: [A-Z][a-z]+)?), (` + strings.Join(sortedKeys(states), "|") +
		`)\b(?:,? (\d{5})(?:-(\d{4}))?\b)?`)
	apartmentPattern    = compile(`\b(Apt|Ste|Bldg)\.? ?#?(\d+[A-Z]?\b|[A-Z]\b)`)
	apartments          = map[string]string{"Apt": "Apartment", "Ste": "Suite", "Bldg": "Building"}
	noBeforePattern     = compile(`\b(?i:no)\. $`)
	stateAfterPattern   = compile(`^(?:$|[.!?;])`)
	headingAfterPattern = compile(`^[.!?;]? [A-Z]{2,}\b`)
)

// streetAddress reads "123 N Main St NW" as "one twenty three North Main Street Northwest".
func streetAddress(m match) (string, bool) {
	houseNumber, letter, direction, name, street, periodMark, afterDirection := m.group(1), m.group(2), m.group(3),
		m.group(4), m.group(5), m.group(6), m.group(7)
	// "No. 10 Downing St." keeps "No." with its number; the later rules read both.
	if noBeforePattern.MatchString(textBefore(m)) {
		return "", false
	}

	words := houseNumberWords(houseNumber)
	if letter != "" {
		words += " " + letter
	}

	if direction != "" {
		words += " " + capitalize(compass[strings.Trim(direction, " .")])
	}

	if expanded, ok := streets[street]; ok {
		street = expanded
	}
	words += " " + name + street
	if afterDirection != "" {
		if strings.HasPrefix(afterDirection, ",") {
			words += ","
		}
		words += " " + capitalize(compass[strings.Trim(afterDirection, ", ")])
	}

	return words + keptPeriod(m, periodMark), true
}

// state reads "Boston, MA 02115" as "Boston, Massachusetts zero two one one five".
func state(m match) (string, bool) {
	city, code, zipCode, zipSuffix := m.group(1), m.group(2), m.group(3), m.group(4)
	if (wordStates[code] && zipCode == "") || isAlphanumeric(nextCharacter(m)) {
		return "", false
	}

	// Without a ZIP code the state must end the sentence ("Boston, MA."); "Rome, MS yet" is a manuscript and "Smith,
	// MD," a doctor, and "Opinion, MS. RETALIATION" is a manuscript before a heading.
	following := textAfter(m)
	if zipCode == "" && (!stateAfterPattern.MatchString(following) || headingAfterPattern.MatchString(following)) {
		return "", false
	}

	words := city + ", " + states[code]
	if zipCode != "" {
		words += " " + digitWords(zipCode)
		if zipSuffix != "" {
			words += " dash " + digitWords(zipSuffix)
		}
	}
	return words, true
}

func apartment(m match) string {
	return apartments[m.group(1)] + " " + m.group(2)
}

var (
	oldStates = map[string]string{"Va": "Virginia", "Mass": "Massachusetts", "Penn": "Pennsylvania",
		"Penna": "Pennsylvania", "Conn": "Connecticut", "Md": "Maryland", "Del": "Delaware", "Ga": "Georgia",
		"Ky": "Kentucky", "Tenn": "Tennessee", "Ala": "Alabama", "Miss": "Mississippi", "Mo": "Missouri",
		"Ill": "Illinois", "Wis": "Wisconsin", "Mich": "Michigan", "Cal": "California", "Calif": "California",
		"Pa": "Pennsylvania", "Vt": "Vermont", "Me": "Maine", "La": "Louisiana", "Tex": "Texas", "Fla": "Florida",
		"Ark": "Arkansas", "Ind": "Indiana", "Kan": "Kansas", "Neb": "Nebraska", "Minn": "Minnesota", "Ore": "Oregon",
		"Colo": "Colorado", "Col": "Colorado", "N.Y": "New York", "N.H": "New Hampshire", "N.J": "New Jersey",
		"R.I": "Rhode Island", "S.C": "South Carolina", "N.C": "North Carolina", "D.C": "D C"}
	oldStateNames   = longestFirst(sortedKeys(oldStates))
	oldStatePattern = compile(`\b([A-Z][a-z]+), (` + oldStateNames + `)\.(?: (\d{5})\b)?`)
	// State abbreviations that are also words, forms of address or titles ('"No, Miss." she said', "Yes, Pa."); these
	// are states only after "in Corinth," and the like, or before a ZIP code.
	wordOldStates = set("Miss", "Pa", "Me", "Mo", "Cal", "Ill", "La", "Ind", "Ore", "Col")
	// A closing quote after a word-like state makes it a form of address ("to Bartram Haugh, Miss.'").
	addressQuoteAfterPattern = compile(`^[\"'\x{2019}\x{201d}]`)
	// "at Glenwood Springs, Col." puts up to one more word of the town's name after the place word.
	placeBeforePattern = compile(`\b(?i:in|at|of|near|from|to) (?:[A-Z][a-z]+ )?$`)
)

// oldState reads "Corinth, Miss. 38834" as "Corinth, Mississippi three eight eight three four".
func oldState(m match) (string, bool) {
	city, state, zipCode := m.group(1), m.group(2), m.group(3)
	following := textAfter(m)
	if letterPeriodPattern.MatchString(following) {
		return "", false
	}

	place := placeBeforePattern.MatchString(textBefore(m)) && !addressQuoteAfterPattern.MatchString(following)
	if wordOldStates[state] && zipCode == "" && !place {
		return "", false
	}

	if zipCode != "" {
		return city + ", " + oldStates[state] + " " + digitWords(zipCode), true
	}
	return city + ", " + oldStates[state] + sentencePeriod(m.text, m.end()), true
}
