package text

import (
	"strings"
)

var (
	// Underscores that mark italics ("_January 29_"); one inside a name ("report_final.pdf") is kept.
	italicPattern = compile(`(^|[^A-Za-z0-9])_+|_+([^A-Za-z0-9]|$)`)
	// A footnote mark set against a word or punctuation ("the walls[3] stood", "Rome.[a]") is not read.
	footnotePattern       = compile(`([A-Za-z.,;:!?'")])(?:\[(?:\d{1,3}|[a-z*])\])+`)
	multipleSpacesPattern = compile(` {2,}`)
)

// typography writes typography in its plain form, names symbols, and reads notation that holds digits before any
// number rule: a note's sharp, key chords, multipliers, ISBNs and a dash glued to a new sentence.
func typography(text string) string {
	text = italicPattern.ReplaceAllString(typographyForRules(text), "$1$2")
	text = earlySymbolWords(meridiemLetters(text))
	text = slashLabelPattern.ReplaceAllString(text, "$1 N")
	text = isbnPattern.replaceFunc(text, isbn)
	text = gluedDashPattern.replaceWhere(text, gluedDash)
	text = sharpPattern.ReplaceAllString(text, "$1 sharp")
	text = techNamePattern.replaceFunc(text, func(m match) string { return techNames[m.group(1)] })
	text = keyChordPattern.replaceFunc(text, keyChord)
	return multiplierPattern.replaceFunc(text, multiplier)
}

// datedPlaces reads what the number rules would split: international phone numbers, abbreviated months, footnote
// marks and old state names.
func datedPlaces(text string) string {
	monthsListed := monthList(text)
	text = internationalPhonePattern.replaceFunc(text, internationalPhone)
	text = monthYearPattern.replaceFunc(text, func(m match) string {
		return monthAbbreviations[m.group(1)] + " " + m.group(2)
	})
	text = monthAlonePattern.replaceWhere(text, monthAlone)
	if monthsListed {
		text = monthAbbreviationPattern.replaceFunc(text, monthInList)
	}

	text = footnotePattern.ReplaceAllString(text, "$1")
	return oldStatePattern.replaceWhere(text, oldState)
}

// numberShapes reads the numbers whose shape decides their reading before any range or date rule sees them: powers,
// decimals, eras, large fractions, tickers and coordinates.
func numberShapes(text string) string {
	text = threadNumberPattern.replaceWhere(text, threadNumber)
	text = labeledPartPattern.replaceWhere(text, labeledPart)
	text = ratingPattern.replaceWhere(text, rating)
	text = countScalePattern.replaceWhere(text, countScale)
	text = shortSecondsPattern.replaceWhere(text, shortSeconds)
	text = statusCodePattern.replaceWhere(text, statusCode)
	text = flightPattern.replaceFunc(text, flight)
	text = parenLetterPattern.replaceFunc(text, func(m match) string {
		return houseNumberWords(m.group(1)) + " " + m.group(2)
	})
	text = teamNumberPattern.replaceFunc(text, teamNumber)
	text = aircraftPattern.replaceFunc(text, aircraft)
	text = powerPattern.replaceFunc(text, power)
	text = indianGroupingPattern.replaceFunc(text, func(m match) string {
		return strings.ReplaceAll(m.group(0), ",", "")
	})
	text = centsOnlyPattern.ReplaceAllString(text, "${1}0.$2")
	text = leadingDecimalPattern.replaceFunc(text, leadingDecimal)
	text = eraYearPattern.replaceWhere(text, eraYear)
	text = largeFractionPattern.replaceWhere(text, largeFraction)
	text = slashFractionUnitPattern.replaceWhere(text, slashFractionUnit)
	text = starProductPattern.ReplaceAllString(text, "$1 times $2")
	text = strings.ReplaceAll(text, "*", "")
	text = rupeesPattern.ReplaceAllString(text, "\u20b9$1")
	text = tickerPattern.replaceFunc(text, ticker)
	return coordinatePattern.replaceFunc(text, coordinate)
}

// placesAndReferences reads street addresses, states, legal and Bible references, money ranges, Latin and number
// abbreviations, before the identifier readers take their digits.
func placesAndReferences(text string) string {
	text = streetAddressPattern.replaceWhere(text, streetAddress)
	text = statePattern.replaceWhere(text, state)
	text = apartmentPattern.replaceFunc(text, apartment)
	text = sectionSignPattern.replaceFunc(text, sectionSign)
	text = statutePattern.replaceFunc(text, statute)
	text = journalIssuePattern.replaceFunc(text, journalIssue)
	text = versusPattern.replaceWhere(text, versus)
	text = numberedBookPattern.replaceFunc(text, numberedBook)
	text = moneyPerPattern.replaceFunc(text, func(m match) string {
		period, ok := perPeriods[m.group(2)]
		if !ok {
			period = m.group(2)
		}
		return m.group(1) + " per " + period
	})
	text = moneyRangePattern.replaceWhere(text, moneyRange)
	text = feetInchesPattern.replaceFunc(text, feetInches)
	text = latinPattern.replaceFunc(text, latin)
	text = spelledInitialismPattern.replaceWhere(text, spelledInitialism)
	return numberAbbreviationPattern.replaceWhere(text, numberAbbreviation)
}

// identifiers reads addresses, handles, versions, phones, serials, codes and hyphened numbers, each by the shape of
// its token, before the date and time rules see their digits.
func identifiers(text string) string {
	// One pass reads one flag per command, since each match ends at its flag.
	for range flagsPerCommand {
		text = commandFlagPattern.replaceWhere(text, commandFlag)
	}
	text = queryPattern.replaceWhere(text, withoutQuery)
	text = localHostPattern.replaceWhere(text, localHost)
	text = addressPattern.replaceWhere(text, addressWords)
	text = mentionPattern.replaceFunc(text, mention)
	text = dottedDatePattern.replaceWhere(text, dottedDate)
	text = dottedPhonePattern.replaceFunc(text, dottedPhone)
	text = versionPattern.replaceWhere(text, version)
	text = dottedTimeRangePattern.replaceWhere(text, dottedTimeRange)
	text = dottedTimePattern.replaceWhere(text, dottedTime)
	text = nanpPhonePattern.replaceFunc(text, nanpPhone)
	text = digitGroupsPattern.replaceWhere(text, digitGroups)
	text = hashNumberPattern.replaceWhere(text, hashNumber)
	text = hashtagPattern.replaceFunc(text, hashtag)
	text = snakeCasePattern.replaceFunc(text, snakeCase)
	text = serialPattern.replaceWhere(text, serial)
	text = alphanumericCodePattern.replaceWhere(text, alphanumericCode)
	text = timeRangePattern.replaceWhere(text, timeRange)
	text = hyphenedNumbersPattern.replaceWhere(text, hyphenedNumbers)
	text = sizeRangePattern.replaceWhere(text, sizeRange)
	return keywordDigitsPattern.replaceWhere(text, keywordDigits)
}

// datesTimesAndMoney reads slashed and colon numbers (dates day first when dayFirst), clock times, money and degrees.
func datesTimesAndMoney(text string, dayFirst bool) string {
	text = slashedNumbersPattern.replaceWhere(text, func(m match) (string, bool) { return slashedNumbers(m, dayFirst) })
	text = colonNumbersPattern.replaceWhere(text, colonNumbers)
	twentyFourHour := twentyFourHourPattern.MatchString(text)
	text = clockTimePattern.replaceWhere(text, func(m match) (string, bool) { return clockTime(m, twentyFourHour) })
	text = militaryTimePattern.replaceFunc(text, militaryTime)
	text = meridiemTimePattern.replaceWhere(text, meridiemTime)
	text = minusMoneyPattern.ReplaceAllString(text, "${1}minus $2")
	text = codeAmountPattern.replaceWhere(text, func(m match) (string, bool) { return currencyCode(m, true) })
	text = amountCodePattern.replaceWhere(text, func(m match) (string, bool) { return currencyCode(m, false) })
	text = currencyNounPattern.replaceFunc(text, func(m match) string {
		return m.group(1) + currencyCodes[m.group(2)].major.plural
	})
	text = moneyPattern.replaceWhere(text, money)
	text = centsPattern.replaceFunc(text, cents)
	return degreesPattern.replaceFunc(text, degrees)
}

// operators reads math operators between their operands, after the dates, times and money that hold the same marks.
func operators(text string) string {
	text = byPattern.ReplaceAllString(text, "$1 by $2")
	// An equals sign makes the whole stretch an equation, so its spaced "x" and "-" are operators ("3 x 4 = 12").
	spacedBy := "$1 by $2"
	if strings.Contains(text, "=") {
		spacedBy = "$1 times $2"
	}

	text = spacedByPattern.ReplaceAllString(text, spacedBy)
	text = variableComparisonPattern.replaceFunc(text, variableComparison)
	// variableComparison may have read every "=" as words ("x <= 3"), and a spaced "-" is minus only in what is
	// still an equation.
	if strings.Contains(text, "=") {
		text = spacedMinusPattern.ReplaceAllString(text, "$1 minus $2")
	}

	text = spacedDividePattern.ReplaceAllString(text, "$1 divided by $2")
	text = dimensionsPattern.replaceWhere(text, dimensions)
	text = dividePattern.ReplaceAllString(text, "$1 by ")
	return symbolForms.read(text)
}

// measures reads fractions and units after the money and time rules, so "$5/hr" and "5:30 pm" are no units.
func measures(text string) string {
	text = fractionUnitPattern.replaceFunc(text, fractionUnit)
	text = fractionPattern.replaceFunc(text, fraction)
	text = unitAdjectivePattern.replaceWhere(text, unitAdjective)
	text = measureUnitPattern.replaceWhere(text, measureUnit)
	return superscriptPattern.replaceFunc(text, superscriptWords)
}

// wordsAndNumerals reads days, titles, word abbreviations and Roman numerals, after the units so "5 min." is a unit.
func wordsAndNumerals(text string) string {
	daysListed := dayList(text)
	text = dayPattern.replaceWhere(text, func(m match) (string, bool) { return day(m, daysListed) })
	text = abbreviationWordsPattern.replaceWhere(text, abbreviationWords)
	text = titleWithoutPeriodPattern.replaceFunc(text, func(m match) string {
		return titles[m.group(1)] + " " + m.group(2)
	})
	text = romanNumeralPattern.replaceWhere(text, romanNumeral)
	return romanCenturyPattern.replaceWhere(text, romanCentury)
}

// numbersAsWords reads every number left, as a date, ordinal, decade, year or cardinal.
func numbersAsWords(text string) string {
	text = spaceBeforeNumberPattern.ReplaceAllString(text, "$1 $2")
	text = pairedNumberPattern.replaceWhere(text, pairedNumber)
	text = spaceAfterNumberPattern.ReplaceAllString(text, "$1 $2")
	text = dayMonthPattern.replaceWhere(text, dayMonth)
	text = monthDayPattern.replaceWhere(text, monthDay)
	text = ordinalPattern.replaceWhere(text, ordinal)
	text = decadePattern.replaceFunc(text, decade)
	text = yearNumberPattern.replaceWhere(text, yearNumber)
	text = negativePattern.ReplaceAllString(text, "${1}minus $2")
	text = colonBetweenNumbersPattern.ReplaceAllString(text, "$1, $2")
	return cardinalPattern.replaceFunc(text, cardinal)
}

// writtenForms reads one stretch of text between sound tags with every number, symbol and structured form (codes,
// addresses, versions, dates, times, money, units, abbreviations) as the words a reader says. Each stage runs before
// the stages that would misread its forms. It reads text as the training data's spoken forms did with their default
// switches, Roman numerals and abbreviations as words and Latin abbreviations by meaning, except rules only old books
// need.
func writtenForms(text string) string {
	written := text
	text = typography(chatWords(markdownPlain(text)))
	dayFirst := slashedDayFirst(text)
	text = datedPlaces(text)
	text = numberShapes(text)
	text = placesAndReferences(text)
	text = identifiers(text)
	text = datesTimesAndMoney(text, dayFirst)
	text = operators(text)
	text = measures(text)
	text = wordsAndNumerals(text)
	text = numbersAsWords(text)
	text = leftoverSymbolWords(text)
	text = multipleSpacesPattern.ReplaceAllString(text, " ")
	// A symbol named at either end of the text must not leave a space there.
	if written == strings.Trim(written, " ") {
		return strings.Trim(text, " ")
	}
	return text
}
