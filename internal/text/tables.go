package text

import (
	"maps"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"
)

func set(words ...string) map[string]bool {
	m := make(map[string]bool, len(words))
	for _, word := range words {
		m[word] = true
	}
	return m
}

func sortedKeys[V any](m map[string]V) []string {
	return slices.Sorted(maps.Keys(m))
}

// joined reads each item and joins the readings with separator.
func joined(items []string, separator string, read func(string) string) string {
	readings := make([]string, len(items))
	for i, item := range items {
		readings[i] = read(item)
	}
	return strings.Join(readings, separator)
}

func union(sets ...map[string]bool) map[string]bool {
	all := map[string]bool{}
	for _, s := range sets {
		for key := range s {
			all[key] = true
		}
	}
	return all
}

func mapValues(m map[string]string) []string {
	return slices.Collect(maps.Values(m))
}

// longestFirst is an alternation of words, longest first so a longer word wins where two start alike, and words of
// one length kept in their order.
func longestFirst(words []string) string {
	sorted := slices.Clone(words)
	slices.SortStableFunc(sorted, func(a, b string) int {
		return utf8.RuneCountInString(b) - utf8.RuneCountInString(a)
	})
	return joined(sorted, "|", regexp.QuoteMeta)
}
