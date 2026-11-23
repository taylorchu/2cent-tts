package text

import (
	"slices"
	"strings"
	"unicode"
)

func isAlphanumeric(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r)
}

// isUpper and isLower need a cased letter, so a token of digits is neither.
func isUpper(s string) bool {
	return strings.ToUpper(s) == s && strings.ToLower(s) != s
}

func isLower(s string) bool {
	return strings.ToLower(s) == s && strings.ToUpper(s) != s
}

func isTitle(s string) bool {
	return s != "" && s == capitalize(s) && isUpper(s[:1])
}

func isDigits(s string) bool {
	return s != "" && strings.IndexFunc(s, func(r rune) bool { return !unicode.IsDigit(r) }) < 0
}

func isAlpha(s string) bool {
	return s != "" && strings.IndexFunc(s, func(r rune) bool { return !unicode.IsLetter(r) }) < 0
}

func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + strings.ToLower(s[1:])
}

func isASCIILetter(c byte) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
}

func isASCIIDigit(c byte) bool {
	return '0' <= c && c <= '9'
}

func allDigits(parts []string) bool {
	return !slices.ContainsFunc(parts, func(part string) bool { return !isDigits(part) })
}

func containsDigit(s string) bool {
	return strings.ContainsAny(s, "0123456789")
}
