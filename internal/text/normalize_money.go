package text

import (
	"regexp"
	"strings"
)

// moneySigns are the dollar, pound, euro, yen (and yuan), won, rupee, ruble, peso, shekel, naira, bitcoin, hryvnia,
// lira, baht and dong signs.
const moneySigns = "$\u00a3\u20ac\u00a5\u20a9\u20b9\u20bd\u20b1\u20aa\u20a6\u20bf\u20b4\u20ba\u0e3f\u20ab"

// moneySignClass is moneySigns inside a regular expression's character class.
var moneySignClass = regexp.QuoteMeta(moneySigns)

// currency names a currency's major and minor units; a currency without a minor unit in use leaves it empty.
type currency struct{ major, minor noun }

var (
	// moneySign matches one money sign, the real's two-character "R$" among them.
	moneySign         = `(?:R\$|[` + moneySignClass + `])`
	moneyRangePattern = compile(`(` + moneySign + `)(` + numberShape + `)-` + moneySign + `?(` +
		numberShape + `)\b`)
	moneyPerPattern = compile(`(` + moneySign + numberShape + `)/` +
		`(month|year|week|day|hour|night|person|head|item|unit|mo|yr|wk|hr)\b`)
	// A price per period written short ("$9/mo") says the period in full.
	perPeriods        = map[string]string{"mo": "month", "yr": "year", "wk": "week", "hr": hourUnit.singular}
	minusMoneyPattern = compile(`(^|[\s(:])-(` + moneySign + `\d)`)
	rupeesPattern     = compile(`\bRs\.? ?(\d)`)
	currencies        = map[string]currency{
		"$":      {noun{"dollar", "dollars"}, noun{"cent", "cents"}},
		"\u00a3": {noun{"pound", "pounds"}, noun{"penny", "pence"}},
		"\u20ac": {noun{"euro", "euros"}, noun{"cent", "cents"}}, "\u00a5": {major: noun{"yen", "yen"}},
		"\u20a9": {major: noun{"won", "won"}}, "\u20b9": {noun{"rupee", "rupees"}, noun{"paisa", "paise"}},
		"\u20bd": {noun{"ruble", "rubles"}, noun{"kopeck", "kopecks"}},
		"\u20b1": {noun{"peso", "pesos"}, noun{"centavo", "centavos"}},
		"\u20aa": {noun{"shekel", "shekels"}, noun{"agora", "agorot"}},
		"\u20a6": {noun{"naira", "naira"}, noun{"kobo", "kobo"}}, "\u20bf": {major: noun{"bitcoin", "bitcoins"}},
		"\u20b4": {major: noun{"hryvnia", "hryvnias"}}, "\u20ba": {major: noun{"lira", "lira"}},
		"\u0e3f": {major: noun{"baht", "baht"}}, "\u20ab": {major: noun{"dong", "dong"}},
		"R$": {noun{"real", "reais"}, noun{"centavo", "centavos"}},
	}
	dollarCountries = map[string]string{"US": "US", "A": "Australian", "AU": "Australian", "C": "Canadian",
		"CA": "Canadian", "NZ": "New Zealand", "HK": "Hong Kong", "S": "Singapore"}
	currencyCodes     = currencyCodeTable()
	currencyCodeNames = strings.Join(sortedKeys(currencyCodes), "|")
	// A currency code after a preposition is the currency's name ("convert to EUR" as "convert to euros").
	currencyNounPattern = compile(`\b((?i:to|into|in|from|per|than|of) )(` + currencyCodeNames + `)\b`)
	codeAmountPattern   = compile(`\b(` + currencyCodeNames + `) ?(` + numberShape +
		`)(?:( (?:thousand|million|billion|trillion))\b)?`)
	amountCodePattern = compile(`\b(` + numberShape + `)(?:( (?:thousand|million|billion|trillion)))? (` +
		currencyCodeNames + `)\b`)
	// The letters that scale a number. After a money sign each is read ("$5k", "$2bn").
	scales = map[string]string{"k": scaleNames[1], "K": scaleNames[1], "m": scaleNames[2], "M": scaleNames[2],
		"b": scaleNames[3], "B": scaleNames[3], "bn": scaleNames[3]}
	// A count in millions or billions before a plural noun ("3M views", "7B parameters"); "3M Company" is a name and
	// "room 2B was" an apartment. Thousands stay a letter, since people say the "K" ("a 5K race", "10K followers").
	countScalePattern = compile(`\b(\d+(?:\.\d+)?)([MB])( [a-z]+s)\b`)
	moneyPattern      = compile(`(?:\b(US|AU|CA|NZ|HK|A|C|S))?(R\$ ?|[` + moneySignClass + `])(` + numberShape +
		`)(?:( (?:thousand|million|billion|trillion))\b|(bn|[kKmMbB])\b)?`)
	centsPattern = compile(`\b(` + numberShape + `) ?\x{a2}`)
	// "$.50" is fifty cents.
	centsOnlyPattern = compile(`([` + moneySignClass + `])\.(\d\d)\b`)
	// A sum before one of these nouns is singular ("a $20 bill"); "$50 reward" keeps the plural.
	moneyCountNouns = set("bill", "note", "coin", "check", "cheque", "piece", "fee", "fine", "bet", "chip", "token",
		"voucher", "bond")
)

// currencyCode reads "EUR 20" and "30 USD" as "twenty euros" and "thirty US dollars".
func currencyCode(m match, codeFirst bool) (string, bool) {
	code, amount, scale := m.group(1), m.group(2), m.group(3)
	if !codeFirst {
		amount, scale, code = m.group(1), m.group(2), m.group(3)
	}

	if (codeFirst && continuesAfter(m)) || (!codeFirst && continuesBefore(m)) {
		return "", false
	}

	names := currencyCodes[code]
	if scale != "" {
		return numberWords(amount) + scale + " " + names.major.plural, true
	}
	return amountWords(amount, names, false), true
}

// amountWords reads an amount in a currency: "19.99" in dollars as "nineteen dollars and ninety nine cents".
func amountWords(amount string, names currency, beforeNoun bool) string {
	whole, cents, _ := strings.Cut(strings.ReplaceAll(amount, ",", ""), ".")
	if cents != "" && (len(cents) != 2 || names.minor.singular == "") {
		return numberWords(amount) + " " + names.major.plural
	}

	var words []string
	if number(whole) != 0 || cents == "" || number(cents) == 0 {
		name := names.major.singular
		if !beforeNoun {
			name = names.major.counted(whole)
		}
		words = append(words, integerTextWords(whole)+" "+name)
	}

	if cents != "" && number(cents) != 0 {
		words = append(words, integerWords(number(cents))+" "+names.minor.counted(strings.TrimLeft(cents, "0")))
	}
	return strings.Join(words, " and ")
}

// countScale reads "3M views" as "3 million views"; a sum ("$3M") is the money rule's.
func countScale(m match) (string, bool) {
	written, scale, noun := m.group(1), m.group(2), m.group(3)
	if strings.ContainsRune(moneySigns+".,", previousCharacter(m)) || notPluralNouns[strings.TrimSpace(noun)] {
		return "", false
	}
	return written + " " + scales[scale] + noun, true
}

// money reads "$19.99" as "nineteen dollars ninety nine cents", "\u00a5500" as "five hundred yen" and "$1.5M" as
// "one point five million dollars".
func money(m match) (string, bool) {
	country, symbol, amount, scaleWord, scaleLetter := m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
	symbol = strings.TrimSpace(symbol)
	names := currencies[symbol]
	if country != "" {
		if symbol != "$" {
			return "", false
		}
		names = dollarsOf(dollarCountries[country])
	}

	scale := strings.TrimSpace(scaleWord)
	if scale == "" {
		scale = scales[scaleLetter]
	}

	if scale != "" {
		return numberWords(amount) + " " + scale + " " + names.major.plural, true
	}

	beforeNoun := !strings.Contains(amount, ".") && moneyCountNouns[lowerWordAfterPattern.firstGroup(textAfter(m))]
	return amountWords(amount, names, beforeNoun), true
}

// moneyRange reads "$5-10" as "5 to $10", read "five to ten dollars".
func moneyRange(m match) (string, bool) {
	if continuesAfter(m) {
		return "", false
	}
	return m.group(2) + " to " + m.group(1) + m.group(3), true
}

func cents(m match) string {
	return numberWords(m.group(1)) + " " + currencies["$"].minor.counted(m.group(1))
}

// currencyCodeTable reads each currency code as the currency its sign names where one does.
func currencyCodeTable() map[string]currency {
	yuan, krone := currency{major: noun{"yuan", "yuan"}}, currency{major: noun{"krone", "kroner"}}
	return map[string]currency{
		"USD": dollarsOf(dollarCountries["US"]), "CAD": dollarsOf(dollarCountries["CA"]),
		"AUD": dollarsOf(dollarCountries["AU"]), "NZD": dollarsOf(dollarCountries["NZ"]),
		"HKD": dollarsOf(dollarCountries["HK"]), "SGD": dollarsOf(dollarCountries["S"]),
		"EUR": currencies["\u20ac"], "GBP": currencies["\u00a3"], "JPY": currencies["\u00a5"],
		"INR": currencies["\u20b9"], "KRW": currencies["\u20a9"], "RUB": currencies["\u20bd"],
		"MXN": currencies["\u20b1"], "BRL": currencies["R$"],
		"CHF": {noun{"Swiss franc", "Swiss francs"}, noun{"centime", "centimes"}}, "CNY": yuan, "RMB": yuan,
		"SEK": {major: noun{"krona", "kronor"}}, "NOK": krone, "DKK": krone, "ZAR": {major: noun{"rand", "rand"}},
		"PLN": {major: noun{"zloty", "zlotys"}},
	}
}

// dollarsOf is the dollar of a country ("Canadian dollars").
func dollarsOf(country string) currency {
	dollars := currencies["$"]
	dollars.major = noun{country + " " + dollars.major.singular, country + " " + dollars.major.plural}
	return dollars
}
