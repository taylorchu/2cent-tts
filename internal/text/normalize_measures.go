package text

import (
	"strings"
)

// The units more than one rule counts.
var (
	hourUnit   = noun{"hour", "hours"}
	minuteUnit = noun{"minute", "minutes"}
	secondUnit = noun{"second", "seconds"}
	degreeUnit = noun{"degree", "degrees"}
)

var (
	degreesPattern = compile(`(-?)\b(` + numberShape + `) ?\x{b0}(?: ?([CFNSEW])\b(\.?))?`)
	fractionSigns  = map[string]string{"\u00bd": "1/2", "\u2153": "1/3", "\u2154": "2/3", "\u00bc": "1/4",
		"\u00be": "3/4", "\u2155": "1/5", "\u215b": "1/8", "\u215c": "3/8", "\u215d": "5/8", "\u215e": "7/8"}
	fractionPattern = compile(`(\d ?)?([` + strings.Join(sortedKeys(fractionSigns), "") + `])`)
	unitNames       = map[string]noun{
		"km/h": {"kilometer per hour", "kilometers per hour"}, "kph": {"kilometer per hour", "kilometers per hour"},
		"m/s": {"meter per second", "meters per second"}, "mph": {"mile per hour", "miles per hour"},
		"rpm": {"revolution per minute", "revolutions per minute"}, "fps": {"frame per second", "frames per second"},
		"kbps": {"kilobit per second", "kilobits per second"}, "Mbps": {"megabit per second", "megabits per second"},
		"Gbps": {"gigabit per second", "gigabits per second"},
		"MB/s": {"megabyte per second", "megabytes per second"}, "mAh": {"milliamp hour", "milliamp hours"},
		"Ah": {"amp hour", "amp hours"}, "Wh": {"watt hour", "watt hours"}, "kWh": {"kilowatt hour", "kilowatt hours"},
		"MWh": {"megawatt hour", "megawatt hours"}, "GWh": {"gigawatt hour", "gigawatt hours"},
		"W": {"watt", "watts"}, "mW": {"milliwatt", "milliwatts"}, "kW": {"kilowatt", "kilowatts"},
		"MW": {"megawatt", "megawatts"}, "GW": {"gigawatt", "gigawatts"}, "V": {"volt", "volts"},
		"mV": {"millivolt", "millivolts"}, "kV": {"kilovolt", "kilovolts"}, "A": {"amp", "amps"},
		"mA": {"milliamp", "milliamps"}, "\u03a9": {"ohm", "ohms"}, "k\u03a9": {"kilohm", "kilohms"},
		"M\u03a9": {"megohm", "megohms"}, "Hz": {"hertz", "hertz"}, "kHz": {"kilohertz", "kilohertz"},
		"MHz": {"megahertz", "megahertz"}, "GHz": {"gigahertz", "gigahertz"}, "Pa": {"pascal", "pascals"},
		"hPa": {"hectopascal", "hectopascals"}, "kPa": {"kilopascal", "kilopascals"},
		"MPa": {"megapascal", "megapascals"}, "psi": {"pound per square inch", "pounds per square inch"},
		"dB": {"decibel", "decibels"}, "km\u00b2": {"square kilometer", "square kilometers"},
		"m\u00b2": {"square meter", "square meters"}, "cm\u00b2": {"square centimeter", "square centimeters"},
		"m\u00b3": {"cubic meter", "cubic meters"}, "cm\u00b3": {"cubic centimeter", "cubic centimeters"},
		"cc": {"cubic centimeter", "cubic centimeters"}, "KB": {"kilobyte", "kilobytes"},
		"kB": {"kilobyte", "kilobytes"}, "MB": {"megabyte", "megabytes"}, "GB": {"gigabyte", "gigabytes"},
		"TB": {"terabyte", "terabytes"}, "PB": {"petabyte", "petabytes"}, "hp": {"horsepower", "horsepower"},
		"tbsp": {"tablespoon", "tablespoons"}, "tsp": {"teaspoon", "teaspoons"}, "ha": {"hectare", "hectares"},
		"min": minuteUnit, "mins": minuteUnit, "hrs": hourUnit, "hr": hourUnit,
		"h": hourUnit, "sec": secondUnit, "secs": secondUnit,
		"ms": {"millisecond", "milliseconds"}, "\u00b5s": {"microsecond", "microseconds"},
		"ns": {"nanosecond", "nanoseconds"}, "kg": {"kilogram", "kilograms"}, "g": {"gram", "grams"},
		"mg": {"milligram", "milligrams"}, "\u00b5g": {"microgram", "micrograms"}, "ng": {"nanogram", "nanograms"},
		"km": {"kilometer", "kilometers"}, "m": {"meter", "meters"}, "cm": {"centimeter", "centimeters"},
		"mm": {"millimeter", "millimeters"}, "\u00b5m": {"micrometer", "micrometers"},
		"nm": {"nanometer", "nanometers"}, "L": {"liter", "liters"}, "mL": {"milliliter", "milliliters"},
		"ml": {"milliliter", "milliliters"}, "dL": {"deciliter", "deciliters"}, "cL": {"centiliter", "centiliters"},
		"gal": {"gallon", "gallons"}, "qt": {"quart", "quarts"}, "fl oz": {"fluid ounce", "fluid ounces"},
		"lbs": {"pound", "pounds"}, "lb": {"pound", "pounds"}, "oz": {"ounce", "ounces"}, "ft": {"foot", "feet"},
		"in": {"inch", "inches"}, "yd": {"yard", "yards"}, "mi": {"mile", "miles"},
		"kcal": {"kilocalorie", "kilocalories"}, "cal": {"calorie", "calories"}, "kJ": {"kilojoule", "kilojoules"},
		"pt": {"pint", "pints"}, "cwt": {"hundredweight", "hundredweight"}}
	feetInchesPattern = compile(`\b(\d{1,2})'(\d{1,2})"`)
	unitAlternation   = longestFirst(sortedKeys(unitNames))
	// A measure or a sum before one of these nouns is singular ("a 12 oz. can", "a $20 bill"); before any other word
	// it keeps its plural ("the 200l. which I earn").
	measureCountNouns = set("bill", "note", "fee", "fine", "bet", "check", "cheque", "coin", "piece", "can", "bottle",
		"bag", "box", "tank", "pack", "battery", "engine", "drive", "run", "walk", "race", "limit", "zone", "room",
		"wall", "pole", "board", "plank", "rope", "stick", "route", "trip", "journey", "hike", "swim", "jar", "glass",
		"cup", "carton", "tin", "sack", "barrel", "bowl", "pot", "pan", "steak", "bar", "block", "fence", "gap",
		"drop", "fall", "climb", "motor", "bulb", "lamp", "charger", "cable", "screen", "chip", "radio", "prize",
		"reward", "ticket", "gift", "loan", "payment", "raise", "bonus", "purchase", "donation", "deposit", "grant",
		"salary", "tip", "discount", "increase", "cut", "jacket", "dress", "suit", "watch", "ring", "phone", "laptop",
		"machine", "turkey", "fish", "radius", "diameter", "fridge", "font", "ladder", "tower", "boat", "yacht",
		"truck", "trailer", "pipe", "hose", "beam", "post", "tent")
	unitAdjectivePattern = compile(`\b(` + numberShape + `)-(` + unitAlternation + `)\b(\.?)`)
	fractionUnitPattern  = compile(`([\x{bd}\x{bc}\x{be}\x{2153}\x{2154}]) ?(` + unitAlternation + `)\b(\.?)`)
	// "1/2 oz." and "4 1/2 lbs." before a unit.
	slashFractionUnitPattern = compile(`\b(?:(\d+) )?(\d)/(\d{1,2}) ?(` + unitAlternation + `)\b(\.?)`)
	pointSizePattern         = compile(`^ (?:font|type|text|size|bold|italic|print)\b`)
	abbreviatedUnits         = set("min", "mins", "hr", "hrs", "ft", "lb", "lbs", "oz", "in", "sec", "secs", "mi",
		"cwt")
	// Units that are also letters or words; after a label ("Section 3 L.") or the same word ("ha 5 ha") they are
	// not units.
	ambiguousUnits     = set("L", "h", "ha", "m", "A", "V", "W", "mi", "in", "pt")
	measureUnitPattern = compile(`\b(` + numberShape + `) ?(` + unitAlternation + `)(\.?)`)
	// Compass points: a coordinate or temperature says them in lowercase, an address capitalized ("123 N Main St").
	compass = map[string]string{"N": "north", "S": "south", "E": "east", "W": "west", "NE": "northeast",
		"NW": "northwest", "SE": "southeast", "SW": "southwest"}
	// ".41" is "point four one"; before a unit it becomes "0.5" for the unit rule ("0.5 mm").
	leadingDecimalPattern = compile(`(^|[\s(\"'-])\.(\d+)\b`)
	// "1/28,000" and "1/7th".
	largeFractionPattern = compile(`\b(\d{1,3})/(\d{1,3}(?:,\d{3}){1,10}|\d{1,30})(?:st|nd|rd|th)?\b`)

	unitAfterPattern    = compile(`^ ?(?:` + unitAlternation + `)\b`)
	ordinalEndPattern   = compile(`(?:st|nd|rd|th)$`)
	capitalAfterPattern = compile(`^ ?[\"']?[A-Z]`)
)

// degrees reads "-3.5\u00b0C" as "minus three point five degrees Celsius".
func degrees(m match) string {
	minus, written, scale, periodMark := m.group(1), m.group(2), m.group(3), keptPeriod(m, m.group(4))
	sign := ""
	if minus != "" {
		sign = minus
		if strings.ContainsRune(" (:", previousCharacter(m)) {
			sign = "minus "
		}
	}

	scaleName := ""
	switch scale {
	case "C":
		scaleName = " Celsius"
	case "F":
		scaleName = " Fahrenheit"
	case "N", "S", "E", "W":
		scaleName = " " + compass[scale]
	}

	return sign + numberWords(written) + " " + degreeUnit.counted(written) + scaleName + periodMark
}

func feetInches(m match) string {
	feet, inches := m.group(1), m.group(2)
	return feet + " " + unitNames["ft"].counted(feet) + " " + inches + " " + unitNames["in"].counted(inches)
}

// fraction reads "\u00bd" as "one half" and "1\u00bd" as "1 and a half".
func fraction(m match) string {
	if whole := m.group(1); whole != "" {
		return strings.TrimSpace(whole) + " and " + signWords(m.group(2), true)
	}
	return signWords(m.group(2), false)
}

// signWords reads a fraction sign as fractionWords reads its numerator and denominator.
func signWords(sign string, afterWhole bool) string {
	numerator, denominator, _ := strings.Cut(fractionSigns[sign], "/")
	return fractionWords(numerator, denominator, afterWhole)
}

func beforeCountNoun(following string) bool {
	return measureCountNouns[lowerWordAfterPattern.firstGroup(following)]
}

// unitAdjective reads "12-oz." as "12 ounce", a measure joined to the noun it describes.
func unitAdjective(m match) (string, bool) {
	written, unitName, periodMark := m.group(1), m.group(2), m.group(3)
	if ambiguousUnits[unitName] && periodMark == "" {
		return "", false
	}

	return written + " " + unitNames[unitName].singular + keptPeriod(m, periodMark), true
}

// fractionUnit reads "\u00bd lb." as "half a pound" and "\u00bc mile" as "one quarter of a mile".
func fractionUnit(m match) string {
	written, unitName, periodMark := m.group(1), m.group(2), m.group(3)
	singular := unitNames[unitName].singular
	words := signWords(written, false) + " of " + article(singular) + " " + singular
	if written == "\u00bd" {
		words = "half " + article(singular) + " " + singular
	}

	return words + keptPeriod(m, periodMark)
}

// slashFractionUnit reads "1/2 oz." as "half an ounce" and "4 1/2 lbs." as "4 and a half pounds".
func slashFractionUnit(m match) (string, bool) {
	whole, numerator, denominator, unitName, periodMark := m.group(1), m.group(2), m.group(3), m.group(4),
		keptPeriod(m, m.group(5))
	names := unitNames[unitName]
	proper := 0 < number(numerator) && number(numerator) < number(denominator)
	if !proper {
		return "", false
	}

	if whole != "" {
		return whole + " and " + fractionWords(numerator, denominator, true) + " " + names.plural + periodMark, true
	}

	if number(numerator) == 1 && number(denominator) == 2 {
		return "half " + article(names.singular) + " " + names.singular + periodMark, true
	}
	singular := article(names.singular) + " " + names.singular
	return fractionWords(numerator, denominator, false) + " of " + singular + periodMark, true
}

// leadingDecimal reads ".41" as "point four one"; before a unit, "0.5" for the unit rule.
func leadingDecimal(m match) string {
	space, digits := m.group(1), m.group(2)
	if unitAfterPattern.MatchString(textAfter(m)) {
		return space + "0." + digits
	}
	return space + "point " + digitWords(digits)
}

// largeFraction reads "1/28,000" as "one twenty eight thousandth" and "1/7th" as "one seventh".
func largeFraction(m match) (string, bool) {
	numerator, denominator := m.group(1), m.group(2)
	if !strings.Contains(denominator, ",") && !ordinalEndPattern.MatchString(m.group(0)) {
		return "", false
	}

	denominator = strings.ReplaceAll(denominator, ",", "")
	if number(numerator) == 0 || compareNumbers(numerator, denominator) >= 0 {
		return "", false
	}
	return fractionWords(numerator, denominator, false), true
}

// measureUnit reads "5 kg" as "five kilograms" and "1 in." as "one inch"; a letter unit after a label ("Section 3 L.")
// or its own word ("ha 5 ha") stays as written, and before a counted noun the unit is singular ("a 5 kg bag").
func measureUnit(m match) (string, bool) {
	written, unitName, periodMark := m.group(1), m.group(2), m.group(3)
	following := textAfter(m)
	if unitInsideWord(m) || (unitName == "in" && periodMark == "") ||
		(periodMark != "" && unitPeriodAbbreviates(m, unitName, following)) || unitAfterLabel(m, unitName) ||
		((unitName == "A" || unitName == "V" || unitName == "W") && capitalAfterPattern.MatchString(following)) {
		return "", false
	}

	return written + " " + unitWord(written, unitName, following) + keptPeriod(m, periodMark), true
}

// unitWord is the unit's name for the number written, singular before a counted noun.
func unitWord(written, unitName, following string) string {
	names := unitNames[unitName]
	// Type is measured in points ("12 pt font").
	pointSize := unitName == "pt" && pointSizePattern.MatchString(following)
	if pointSize {
		names = noun{"point", "points"}
	}

	name := names.counted(written)
	if pointSize || beforeCountNoun(following) {
		name = names.singular
	}
	return name
}

// unitInsideWord is whether a unit's number or name runs into a word, an amount or a path ("5kgs", "$5 m", "5 m/").
func unitInsideWord(m match) bool {
	previous, next := previousCharacter(m), nextCharacter(m)
	return isAlphanumeric(previous) || strings.ContainsRune(moneySigns+".,", previous) || isAlphanumeric(next) ||
		next == '/' || next == '\''
}

// unitPeriodAbbreviates is whether the period after a unit belongs to a book's abbreviation, not to the unit.
func unitPeriodAbbreviates(m match, unitName, following string) bool {
	// An abbreviated unit ("min.") takes its period anywhere, but after a one-letter unit a period inside a sentence
	// marks a book's abbreviation ("4 h. 30 m.") unless a comma follows, and a longer one may end a sentence.
	return !abbreviatedUnits[unitName] && strings.TrimSpace(following) != "" &&
		!strings.HasPrefix(following, ",") && !strings.HasPrefix(following, ";") &&
		(ambiguousUnits[unitName] || !sentenceEnd(m.text, m.end()))
}

// unitAfterLabel is whether a letter unit follows a label ("Section 3 L.") or its own word ("ha 5 ha").
func unitAfterLabel(m match, unitName string) bool {
	previousWord, ok := wordBeforePattern.find(textBefore(m))
	if !ok || !ambiguousUnits[unitName] {
		return false
	}

	word := strings.ToLower(previousWord.group(1))
	return labelWords[word] || strings.EqualFold(word, unitName)
}
