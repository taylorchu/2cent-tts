package text

import (
	"slices"
	"strings"
)

const addressLabel = `[A-Za-z0-9_][A-Za-z0-9_+-]{0,62}`

const hostLabel = `[A-Za-z0-9_][A-Za-z0-9_-]{0,62}`

var (
	// "P/N" and "S/N" before a part or serial number are said as letters.
	slashLabelPattern = compile(`\b([PS])/N\b`)
	// An ISBN reads its digits one by one, its hyphens silent ("ISBN 978-0-306-40615-7").
	isbnPattern = compile(`\bISBN(?:-1[03])?:? ?((?:\d[- ]?){9,12}[\dXx])\b`)
	// A capitals-and-digits token that switches between letters and digits at least twice is a serial or VIN, read
	// character by character ("1HGCM82633A004352").
	serialPattern = compile(`\b[A-Z0-9]{6,30}\b`)
	// Elements with a one-letter symbol: a token of these and digits is a chemical formula ("C6H12O6"), not a serial.
	oneLetterElements = "HBCNOFPSKVYIWU"
	// A journal's volume, issue and page range as a reference list writes them ("52(1), 1-12").
	journalIssuePattern = compile(`\b(\d{1,4})\((\d{1,5})\)[,:] ?(\d{1,6}) ?- ?(\d{1,6})\b`)
	// A statute section with a dot reads both parts in pairs ("section 1910.120" as "section nineteen ten point one
	// twenty").
	statutePattern = compile(`\b((?i:sections?|C\.F\.R\.|CFR) )(\d{3,4})\.(\d{1,4})\b`)
	// Technical names whose spelling misleads, and tool and format names a developer spells, written in capitals.
	techNames = map[string]string{"nginx": "engine X", "kubectl": "cube control", "npm": "NPM", "npx": "NPX",
		"ssh": "SSH", "api": "API", "url": "URL", "gpu": "GPU", "cpu": "CPU", "ssd": "SSD", "usb": "USB", "vpn": "VPN",
		"dns": "DNS", "aws": "AWS", "html": "HTML", "css": "CSS", "csv": "CSV"}
	techNamePattern = compile(`\b(` + longestFirst(sortedKeys(techNames)) + `)\b`)
	// The words of a handle or hashtag: runs of capitalized or lowercase letters, capitals, or digits.
	namePartPattern = compile(`[A-Z]+|[a-z]+|\d+`)
	addressPattern  = compile(`(?:(?:https?|s?ftp)://|/)?` + addressLabel + repeated(`(?:[./@]`+addressLabel+`)`, 1,
		20))
	// Only these endings make "word.word" a domain, since book text sometimes drops the space after a period
	// ("end.he").
	domainEndings = set("com", "org", "net", "edu", "gov", "io", "co", "uk", "sh", "ly", "ai", "app", "dev", "info",
		"biz", "tv", "fm", "de", "fr", "nl", "eu", "ca", "au", "ch", "jp", "xyz")
	genericDomainEndings = set("com", "org", "net", "edu", "gov", "io", "ai", "app", "dev", "info", "biz", "xyz")
	fileExtensions       = set("pdf", "txt", "png", "jpg", "jpeg", "gif", "svg", "webp", "csv", "xls", "xlsx", "doc",
		"docx", "ppt", "pptx", "zip", "gz", "tar", "rar", "exe", "dmg", "apk", "sh", "py", "js", "json", "xml",
		"yaml", "yml", "toml", "ini", "cfg", "conf", "log", "md", "html", "htm", "css", "mp3", "mp4", "wav", "mov",
		"sql", "db", "bak", "iso")
	addressSymbols = map[string]string{".": "dot", "/": "slash", "@": "at", "_": "underscore", "-": "dash",
		"+": "plus"}
	// Parts of an address said as written although the spelling rules below would spell them.
	saidAsWritten    = set("app", "org", "info", "conf", "co")
	addressPartWords = map[string]string{"json": "jason", "jpg": "J peg", "jpeg": "J peg"}
	twoLetterWords   = set("no", "my", "go", "me", "to", "do", "up", "in", "on", "at", "we", "be", "so", "it", "is",
		"of", "or", "if", "an", "as", "by", "he", "hi")
	// A three-letter part is said as a word only when it sounds like one ("log", "zip"); otherwise it is spelled.
	consonantVowelConsonantPattern = compile(`^[b-df-hj-np-tv-z][aeiou][b-df-hj-np-tv-z]$`)
	hashNumberPattern              = compile(`#([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)`)
	alphanumericCodePattern        = compile(`\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b`)
	hashtagPattern                 = compile(`(^|\s)#(\d*[A-Za-z][A-Za-z0-9_]*)`)
	// A code name joined by underscores ("user_id"), with the empty call parentheses of a function ("get_user()").
	snakeCasePattern = compile(`\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?:\(\))?`)
	// A local host and its port ("http://localhost:3000"), which has no dot for the address rule to split at.
	localHostPattern = compile(`\b(?:https?://)?(localhost|127\.0\.0\.1)(?::(\d{2,5}))?\b`)
	// A web address with its query or fragment ("example.com/unsubscribe?id=123"); readers stop before the "?" or
	// "#".
	queryPattern = compile(`((?:https?://)?` + hostLabel + repeated(`(?:\.`+hostLabel+`)`, 1, 20) +
		`(?:/[A-Za-z0-9_.+%/-]{0,200})?)[?#][A-Za-z0-9_=&%+./-]{0,500}[A-Za-z0-9_=&%+/-]`)
	// A command-line flag a few words after a command ("git rebase -i", "ls -la"), said "dash" and its letters; book
	// text sets a dash before a word ("tree -top"), so a flag needs the command.
	commandFlagPattern = compile(`\b((?:git|ls|rm|cp|mv|mkdir|grep|find|npm|npx|yarn|pip|pip3|python|python3|node` +
		`|docker|curl|wget|ssh|scp|tar|chmod|chown|sudo|brew|apt|cargo|make|cat|head|tail|sort|du|df|ps|kill|ping` +
		`|unzip|zip|rsync|ln|touch|conda|gh)(?: [A-Za-z0-9_./~:=@-]+){0,4}? )-([a-z]{1,3})\b`)
	// A North American phone number written with hyphens ("1-800-555-0199"), read as every phone number is.
	hyphenedPhonePattern = compile(`^(?:1-)?\d{3}-\d{3}-\d{4}$`)
	mentionPattern       = compile(`(^|[\s(])@([A-Za-z0-9_]+)\b`)
	// A ticker after "$" or an exchange, with its share class after a dot ("$BRK.A").
	tickerPattern = compile(`\$([A-Z]{1,5}(?:\.[A-Z])?)\b|\b(NASDAQ|NYSE|AMEX|LSE|TSX): ?([A-Z]{1,5}(?:\.[A-Z])?)\b`)

	identifierSeparatorPattern = compile(`[-/ ]`)
	runPattern                 = compile(`[A-Za-z]+|\d+`)
	digitsAndSeparatorsPattern = compile(`^[\d/ -]+$`)
	capitalsOrDigitsPattern    = compile(`[A-Z]+|\d+`)
	vowelPattern               = compile(`[aeiou]`)
	addressPiecePattern        = compile(`[./@_+-]|[^./@_+-]+`)
	codeAfterDashPattern       = compile(`^[-.][A-Za-z0-9]`)
	lettersPattern             = compile(`[A-Za-z]+`)
)

// flagsPerCommand is the most flags read after one command ("ls -l -a -h").
const flagsPerCommand = 4

// identifierRunWords reads one run of an identifier's letters or digits: a year as a year, a short number whole
// unless oneByOne, other digits one by one; a run of four or more letters in one case is a word, said in lowercase
// ("SAVE"), unless oneByOne, and other runs are spelled.
func identifierRunWords(run string, years, oneByOne bool) string {
	if isDigits(run) {
		if years && len(run) == 4 && isYear(number(run)) {
			return yearWords(number(run))
		}

		if !oneByOne && len(run) <= 2 && (run == "0" || run[0] != '0') {
			return smallNumberWords(number(run))
		}
		return digitWords(run)
	}

	if !oneByOne && len(run) >= 4 && (isUpper(run) || isLower(run)) {
		return strings.ToLower(run)
	}
	return spelled(run)
}

// identifierWords reads an identifier such as "AB-4471-Z" as "A B four four seven one Z"; an identifier of digits
// alone pauses at its hyphens, slashes and spaces, which are otherwise silent.
func identifierWords(code string, years, oneByOne bool) string {
	parts := identifierSeparatorPattern.Split(code, -1)
	if hyphenedPhonePattern.MatchString(code) {
		return phoneWords(parts)
	}

	separator := " "
	if digitsAndSeparatorsPattern.MatchString(code) {
		separator = ", "
	}
	return joined(parts, separator, func(part string) string { return identifierPartWords(part, years, oneByOne) })
}

// identifierPartWords reads the letter and digit runs of one part of an identifier.
func identifierPartWords(part string, years, oneByOne bool) string {
	return joined(runPattern.FindAllString(part, -1), " ", func(run string) string {
		return identifierRunWords(run, years, oneByOne)
	})
}

// serial reads a serial or VIN of capitals and digits character by character, and declines for a plainer token
// ("COVID19").
func serial(m match) (string, bool) {
	token := m.group(0)
	if len(capitalsOrDigitsPattern.FindAllString(token, -1)) < 3 || isDigits(token) || isAlpha(token) {
		return "", false
	}

	formula := !strings.ContainsFunc(token, func(character rune) bool {
		return !strings.ContainsRune("0123456789"+oneLetterElements, character)
	})
	if formula {
		return "", false
	}
	return identifierWords(token, false, true), true
}

func isbn(m match) string {
	words := []string{"I S B N"}
	for _, character := range []byte(m.group(1)) {
		switch {
		case character == 'X' || character == 'x':
			words = append(words, "X")
		case isASCIIDigit(character):
			words = append(words, digitNames[character-'0'])
		}
	}
	return strings.Join(words, " ")
}

// journalIssue reads "52(1), 1-12" as "volume 52, issue 1, pages 1 to 12", the numbers left for the number rules.
func journalIssue(m match) string {
	return "volume " + m.group(1) + ", issue " + m.group(2) + ", pages " + m.group(3) + " to " + m.group(4)
}

// statute reads "section 1910.120" as "section nineteen ten point one twenty".
func statute(m match) string {
	label, whole, part := m.group(1), m.group(2), m.group(3)
	return label + houseNumberWords(whole) + " point " + houseNumberWords(part)
}

// nameParts are the words of a handle or hashtag: a capital starts a word, and the last capital of a run before
// lowercase letters starts the next word ("NBAFinals" as "NBA", "Finals").
func nameParts(name string) []string {
	var parts []string
	for _, run := range namePartPattern.FindAllString(name, -1) {
		if isLower(run) && len(parts) > 0 && isUpper(parts[len(parts)-1]) {
			capitals := parts[len(parts)-1]
			parts = parts[:len(parts)-1]
			if len(capitals) > 1 {
				parts = append(parts, capitals[:len(capitals)-1])
			}
			run = capitals[len(capitals)-1:] + run
		}
		parts = append(parts, run)
	}
	return parts
}

// nameWords reads a handle or hashtag split at underscores, case changes and digits ("sales_team" as "sales team",
// "player42" as "player four two"); with numbers its digits stay a number for the number rules ("Election 2026").
func nameWords(name string, numbers bool) string {
	parts := nameParts(name)
	words := make([]string, len(parts))
	for i, part := range parts {
		switch {
		case isDigits(part) && numbers:
			words[i] = part
		case isDigits(part):
			words[i] = digitWords(part)
		default:
			words[i] = addressLetters(part, false)
		}
	}
	return strings.Join(words, " ")
}

// addressLetters reads a run of letters in an address: a word as written, an extension or initialism spelled.
func addressLetters(run string, ending bool) string {
	lower := strings.ToLower(run)
	if saidAsWritten[lower] {
		return run
	}

	if word, ok := addressPartWords[lower]; ok {
		return word
	}

	if !isLower(run) && !isTitle(run) {
		return spelled(run)
	}

	// A three-letter part that starts with a consonant and holds a vowel is said ("doe", "foo"); one that starts
	// with a vowel or holds none is spelled ("api", "www").
	said := len(run) > 3 || (len(run) == 2 && twoLetterWords[lower]) ||
		(len(run) == 3 && !strings.ContainsRune("aeiouy", rune(lower[0])) && vowelPattern.MatchString(lower))
	if ending {
		said = consonantVowelConsonantPattern.MatchString(lower)
	}

	if said {
		return run
	}
	return spelled(run)
}

// isAddress is whether token is a web or email address, a path or a file name: a scheme, an "@" before a host, a
// known file extension, a path of two or more parts, or a host with a generic ending or "www".
func isAddress(token string) bool {
	if slices.ContainsFunc([]string{"http://", "https://", "ftp://", "sftp://"}, func(scheme string) bool {
		return strings.HasPrefix(token, scheme)
	}) {
		return true
	}

	// Extensions and domain endings match in lowercase only, since a scanned book drops the space after a period
	// ("went out.Log", "M.de Treville").
	last := token[strings.LastIndex(token, "/")+1:]
	if dot := strings.LastIndex(last, "."); dot >= 0 && fileExtensions[last[dot+1:]] {
		return true
	}

	host, _, _ := strings.Cut(token, "/")
	if strings.Contains(host, "@") {
		return strings.Contains(strings.Split(host, "@")[1], ".")
	}

	if !strings.HasPrefix(token, "/") && strings.Contains(host, ".") {
		labels := strings.Split(host, ".")
		ending := labels[len(labels)-1]
		// A bare host with a country ending is a scan's missing space ("evo.de", "Mme.de") or initials ("S.W.ly",
		// southwesterly) unless "www" or a path marks it.
		if !strings.Contains(token, "/") && labels[0] != "www" && !genericDomainEndings[ending] {
			return false
		}
		return domainEndings[ending]
	}
	return strings.HasPrefix(token, "/") && strings.Count(token, "/") >= 2
}

// addressWords reads a URL, email address, path or file name out: "notes.txt" as "notes dot T X T".
func addressWords(m match) (string, bool) {
	token := m.group(0)
	if !isAddress(token) {
		return "", false
	}

	var words []string
	if scheme, rest, found := strings.Cut(token, "://"); found {
		// Readers drop a web scheme ("https://example.com" as "example dot com") but say any other.
		if lower := strings.ToLower(scheme); lower != "http" && lower != "https" {
			words = append(words, spelled(scheme)+" colon slash slash")
		}
		token = rest
	}

	pieces := addressPiecePattern.FindAllString(token, -1)
	for i, piece := range pieces {
		if symbol, ok := addressSymbols[piece]; ok {
			words = append(words, symbol)
			continue
		}

		ending := i == len(pieces)-1 && i > 0 && pieces[i-1] == "."
		for _, run := range runPattern.FindAllString(piece, -1) {
			if isDigits(run) {
				words = append(words, addressDigits(run))
			} else {
				words = append(words, addressLetters(run, ending))
			}
		}
	}
	return strings.Join(words, " "), true
}

// addressDigits reads digits in an address one by one, a year as a year ("budget_2027.xlsx").
func addressDigits(run string) string {
	if len(run) == 4 && 1900 <= number(run) && number(run) <= 2099 {
		return yearWords(number(run))
	}
	return digitWords(run)
}

// hashtag reads "#ThrowbackThursday" as "hashtag Throwback Thursday", its digits a number ("#100DaysOfCode").
func hashtag(m match) string {
	return m.group(1) + "hashtag " + nameWords(m.group(2), true)
}

// snakeCase reads "user_id" as "user id" and "get_user()" as "get user".
func snakeCase(m match) string {
	return strings.ReplaceAll(m.group(1), "_", " ")
}

// withoutQuery is a web address without its query or fragment, or declines when the text before the "?" is no address.
func withoutQuery(m match) (string, bool) {
	if !isAddress(m.group(1)) {
		return "", false
	}
	return m.group(1), true
}

// commandFlag reads "git rebase -i" as "git rebase dash I" and "ls -la" as "ls dash L A"; a dash joined to more text is
// left.
func commandFlag(m match) (string, bool) {
	if nextCharacter(m) == '-' {
		return "", false
	}
	return m.group(1) + "dash " + spelled(m.group(2)), true
}

// localHost reads "localhost:3000" as "localhost colon 3000"; a bare "localhost" or a loopback address without a
// scheme or a port is left to the other rules.
func localHost(m match) (string, bool) {
	host, port := m.group(1), m.group(2)
	if port == "" && !strings.Contains(m.group(0), "://") {
		return "", false
	}

	hostWords := "one two seven dot zero dot zero dot one"
	if host == "localhost" {
		hostWords = "localhost"
	}

	if port != "" {
		return hostWords + " colon " + port, true
	}
	return hostWords, true
}

// mention reads "@sales_team" as "at sales team".
func mention(m match) string {
	return m.group(1) + "at " + nameWords(m.group(2), false)
}

// ticker reads "$AAPL", "NASDAQ: AAPL" and "$BRK.A" with the ticker spelled and its class after "dot".
func ticker(m match) string {
	symbol, exchange, listed := m.group(1), m.group(2), m.group(3)
	written := symbol
	if written == "" {
		written = listed
	}

	spoken := joined(strings.Split(written, "."), " dot ", spelled)
	if symbol != "" {
		return spoken
	}
	return exchange + " " + spoken
}

// hashNumber reads "#12" as "number 12"; a longer number or a code after "#" is read part by part.
func hashNumber(m match) (string, bool) {
	token := m.group(1)
	if isAlphanumeric(previousCharacter(m)) || !containsDigit(token) || strings.ToUpper(token) != token {
		return "", false
	}

	switch {
	case isDigits(token) && len(token) <= 3:
		return "number " + token, true
	case isDigits(token):
		return "number " + digitWords(token), true
	}
	return "number " + identifierWords(token, lettersPattern.MatchString(token), false), true
}

// alphanumericCode reads a hyphenated code of capitals and digits ("AB-4471-Z") part by part instead of as large
// numbers.
func alphanumericCode(m match) (string, bool) {
	code := m.group(0)
	letterRuns := lettersPattern.FindAllString(code, -1)
	if len(letterRuns) == 0 || !containsDigit(code) || strings.ToLower(code) == code {
		return "", false
	}

	if slices.ContainsFunc(letterRuns, func(run string) bool { return !isUpper(run) && len(run) > 2 }) {
		return "", false
	}

	if strings.ContainsRune("./#-", previousCharacter(m)) || codeAfterDashPattern.MatchString(textAfter(m)) {
		return "", false
	}
	return identifierWords(code, true, false), true
}
